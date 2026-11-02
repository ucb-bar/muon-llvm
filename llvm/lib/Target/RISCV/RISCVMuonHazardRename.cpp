//===- RISCVMuonHazardRename.cpp - Rename regs to avoid WAR/WAW stalls ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Muon core does not admit an instruction into its reservation station
// while the destination register has a pending write (WAW) or a pending read
// by an unissued older instruction (WAR). Admission is in order per warp, so
// an instruction that reuses a recently touched register blocks every younger
// instruction of the warp. The register allocator reuses a register right
// after its last read, which turns most straight-line code into one serial
// chain.
//
// This pass renames block-local live ranges after register allocation. It
// walks each block in program order and gives each renamable live range the
// free register that was least recently read or written. It does not add or
// move instructions.
//
// A live range is renamable when all of its operands are renamable, untied,
// explicit and not early-clobber, and it is neither live into nor out of the
// block. A candidate register must be allocatable, of every register class
// that the operands require, not clobbered by a call inside the range, and
// either caller-saved or saved by this function.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
#define DEBUG_TYPE "riscv-muon-hazard-rename"
#define RISCV_MUON_HAZARD_RENAME_NAME "RISC-V Muon hazard-aware register renaming"

STATISTIC(NumRangesRenamed, "Number of live ranges renamed");
STATISTIC(NumBlocksSkipped, "Number of blocks left unchanged");

static cl::opt<bool> EnableMuonHazardRename(
    "muon-hazard-rename", cl::Hidden, cl::init(false),
    cl::desc("Rename block-local live ranges to the least recently used "
             "free register (Muon WAR/WAW admission stalls)"));

static cl::opt<bool> MuonHazardRenameUsedOnly(
    "muon-hazard-rename-used-only", cl::Hidden, cl::init(false),
    cl::desc("Rename only into registers that the function already writes, "
             "so the set of written registers does not grow"));

static cl::opt<bool> MuonHazardRenameVerify(
    "muon-hazard-rename-verify", cl::Hidden, cl::init(true),
    cl::desc("Check that every read and every live-out register sees the "
             "same value before and after renaming; abort if not"));

// Check falsification only: treat early-clobber results as ordinary results
// (the defect fixed on 2026-10-29), which the self-check must catch.
static cl::opt<bool> MuonHazardRenameBreakEC("muon-hazard-rename-break-ec",
                                             cl::Hidden, cl::init(false));
// Check falsification only: rename values that are live out of the block.
static cl::opt<bool> MuonHazardRenameBreakLiveOut(
    "muon-hazard-rename-break-liveout", cl::Hidden, cl::init(false));

static cl::list<std::string> MuonHazardRenameOnly(
    "muon-hazard-rename-only", cl::Hidden, cl::CommaSeparated,
    cl::desc("Rename only in functions whose name contains one of these"));

static cl::list<std::string> MuonHazardRenameSkip(
    "muon-hazard-rename-skip", cl::Hidden, cl::CommaSeparated,
    cl::desc("Do not rename in functions whose name contains one of these"));

namespace {

// One live range of a physical register inside a block. Start is the index
// of the defining instruction (-1 when live into the block). Begin and Finish
// bound the range in slots: instruction I reads its operands in slot 2I and
// writes its results in slot 2I+1, except early-clobber results, which are
// written in slot 2I (before the instruction has read all of its inputs).
// Finish is the read slot of the last read, the write slot for a value that
// is never read, and 2N when the value is live out of the block. Two ranges
// can share a register only if their slot intervals are disjoint.
struct Range {
  MCPhysReg Reg;
  int Start;
  int Begin;
  int Finish;
  bool Fixed;
  SmallVector<MachineOperand *, 4> Ops;
};

class RISCVMuonHazardRename : public MachineFunctionPass {
public:
  static char ID;

  RISCVMuonHazardRename() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }
  StringRef getPassName() const override {
    return RISCV_MUON_HAZARD_RENAME_NAME;
  }

private:
  const TargetRegisterInfo *TRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  // Registers this pass may assign; indexed by the register number.
  BitVector Candidate;

  bool renameBlock(MachineBasicBlock &MBB);
  void verifyBlock(MachineBasicBlock &MBB, ArrayRef<MachineInstr *> Instrs,
                   const DenseMap<const MachineOperand *, MCPhysReg> &OrigReg,
                   const LivePhysRegs &LiveOut);
};

} // end anonymous namespace

char RISCVMuonHazardRename::ID = 0;
INITIALIZE_PASS(RISCVMuonHazardRename, DEBUG_TYPE,
                RISCV_MUON_HAZARD_RENAME_NAME, false, false)

FunctionPass *llvm::createRISCVMuonHazardRenamePass() {
  return new RISCVMuonHazardRename();
}

static bool isGPR(MCPhysReg Reg) {
  return RISCV::GPRRegClass.contains(Reg);
}

bool RISCVMuonHazardRename::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableMuonHazardRename || skipFunction(MF.getFunction()))
    return false;
  if (!MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::TracksLiveness))
    return false;
  StringRef Name = MF.getName();
  auto matches = [&](const cl::list<std::string> &L) {
    return llvm::any_of(L, [&](const std::string &S) { return Name.contains(S); });
  };
  if (!MuonHazardRenameOnly.empty() && !matches(MuonHazardRenameOnly))
    return false;
  if (matches(MuonHazardRenameSkip))
    return false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();

  // Callee-saved registers are candidates only when this function saves them.
  BitVector CalleeSaved(TRI->getNumRegs());
  for (const MCPhysReg *CSR = MRI->getCalleeSavedRegs(); CSR && *CSR; ++CSR)
    CalleeSaved.set(*CSR);
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (MFI.isCalleeSavedInfoValid())
    for (const CalleeSavedInfo &Info : MFI.getCalleeSavedInfo())
      CalleeSaved.reset(Info.getReg());

  BitVector Written(TRI->getNumRegs());
  if (MuonHazardRenameUsedOnly)
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB)
        for (const MachineOperand &MO : MI.operands())
          if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical())
            Written.set(MO.getReg());

  Candidate = BitVector(TRI->getNumRegs());
  for (MCPhysReg Reg : RISCV::GPRRegClass) {
    if (MRI->isReserved(Reg) || CalleeSaved.test(Reg) || Reg == RISCV::X0)
      continue;
    if (MuonHazardRenameUsedOnly && !Written.test(Reg))
      continue;
    Candidate.set(Reg);
  }

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= renameBlock(MBB);
  return Changed;
}

bool RISCVMuonHazardRename::renameBlock(MachineBasicBlock &MBB) {
  SmallVector<MachineInstr *, 256> Instrs;
  for (MachineInstr &MI : MBB) {
    if (MI.isBundle() || MI.isBundled()) {
      ++NumBlocksSkipped;
      return false;
    }
    if (MI.isDebugInstr())
      continue;
    Instrs.push_back(&MI);
  }
  const int N = Instrs.size();
  if (N < 2)
    return false;

  LivePhysRegs LiveOut(*TRI);
  LiveOut.addLiveOuts(MBB);
  LivePhysRegs LiveIn(*TRI);
  LiveIn.addLiveIns(MBB);

  // Build the live ranges of every GPR in the block.
  std::vector<Range> Ranges;
  DenseMap<MCPhysReg, int> Open; // register -> index of its open range
  // Register -> instruction indices where a call clobbers it.
  DenseMap<MCPhysReg, SmallVector<int, 4>> Clobbers;

  auto openRange = [&](MCPhysReg Reg, int Start, int Begin,
                       bool Fixed) -> Range & {
    Ranges.push_back({Reg, Start, Begin, Begin, Fixed, {}});
    Open[Reg] = Ranges.size() - 1;
    return Ranges.back();
  };

  for (MCPhysReg Reg : RISCV::GPRRegClass)
    if (Reg != RISCV::X0 && !LiveIn.available(*MRI, Reg))
      openRange(Reg, -1, -1, true);

  for (int I = 0; I < N; ++I) {
    MachineInstr &MI = *Instrs[I];
    // A register referenced through a register that is not a plain GPR
    // (a register pair) is out of scope; leave the block unchanged.
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;
      MCPhysReg Reg = MO.getReg();
      if (!isGPR(Reg))
        for (MCPhysReg G : RISCV::GPRRegClass)
          if (TRI->regsOverlap(Reg, G)) {
            ++NumBlocksSkipped;
            return false;
          }
    }

    // Reads extend the open range.
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isUse() || MO.isUndef() || !MO.getReg())
        continue;
      MCPhysReg Reg = MO.getReg();
      if (!isGPR(Reg) || Reg == RISCV::X0)
        continue;
      auto It = Open.find(Reg);
      Range *R;
      if (It == Open.end())
        R = &openRange(Reg, -1, -1, true); // live in but missing from the list
      else
        R = &Ranges[It->second];
      R->Finish = 2 * I;
      R->Ops.push_back(&MO);
      if (!MO.isRenamable() || MO.isTied() || MO.isImplicit())
        R->Fixed = true;
    }

    // Call clobbers end the open range.
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isRegMask())
        continue;
      for (MCPhysReg Reg : RISCV::GPRRegClass)
        if (Reg != RISCV::X0 && MO.clobbersPhysReg(Reg)) {
          Clobbers[Reg].push_back(I);
          Open.erase(Reg);
        }
    }

    // Writes start a new range.
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef() || !MO.getReg())
        continue;
      MCPhysReg Reg = MO.getReg();
      if (!isGPR(Reg) || Reg == RISCV::X0)
        continue;
      const int Begin =
          MO.isEarlyClobber() && !MuonHazardRenameBreakEC ? 2 * I : 2 * I + 1;
      auto It = Open.find(Reg);
      if (It != Open.end() && Ranges[It->second].Start == I) {
        // Second def of the same register in one instruction.
        Range &R = Ranges[It->second];
        R.Ops.push_back(&MO);
        R.Begin = std::min(R.Begin, Begin);
        R.Fixed = true;
        continue;
      }
      Range &R = openRange(Reg, I, Begin, false);
      R.Ops.push_back(&MO);
      if (!MO.isRenamable() || MO.isTied() || MO.isImplicit() ||
          MO.isEarlyClobber() || MO.getSubReg())
        R.Fixed = true;
    }
  }
  for (auto &[Reg, Idx] : Open)
    if (!LiveOut.available(*MRI, Reg)) {
      Ranges[Idx].Finish = 2 * N;
      if (!MuonHazardRenameBreakLiveOut || Ranges[Idx].Start < 0)
        Ranges[Idx].Fixed = true;
    }

  // Required register class of each renamable range.
  std::vector<const TargetRegisterClass *> RC(Ranges.size(), nullptr);
  for (size_t K = 0; K < Ranges.size(); ++K) {
    Range &R = Ranges[K];
    if (R.Fixed)
      continue;
    const TargetRegisterClass *C = &RISCV::GPRRegClass;
    for (MachineOperand *MO : R.Ops) {
      MachineInstr *MI = MO->getParent();
      const TargetRegisterClass *OpRC =
          MI->getRegClassConstraint(MI->getOperandNo(MO), TII, TRI);
      if (!OpRC) {
        R.Fixed = true;
        break;
      }
      C = TRI->getCommonSubClass(C, OpRC);
      if (!C) {
        R.Fixed = true;
        break;
      }
    }
    RC[K] = C;
  }

  // Occupancy and reference times per register. Fixed ranges and call
  // clobbers are placed first; renamed ranges are added as they are assigned.
  const unsigned NumRegs = TRI->getNumRegs();
  std::vector<SmallVector<std::pair<int, int>, 8>> Busy(NumRegs);
  std::vector<SmallVector<int, 16>> Refs(NumRegs);
  // Map each instruction to its index for reference times.
  DenseMap<const MachineInstr *, int> Index;
  for (int I = 0; I < N; ++I)
    Index[Instrs[I]] = I;
  auto addRefs = [&](MCPhysReg Reg, const Range &R) {
    Busy[Reg].push_back({R.Begin, R.Finish});
    for (MachineOperand *MO : R.Ops)
      Refs[Reg].push_back(Index[MO->getParent()]);
  };

  bool AnyRenamable = false;
  for (const Range &R : Ranges) {
    if (R.Fixed)
      addRefs(R.Reg, R);
    else
      AnyRenamable = true;
  }
  if (!AnyRenamable)
    return false;
  for (auto &[Reg, Idx] : Clobbers)
    for (int I : Idx) {
      Busy[Reg].push_back({2 * I + 1, 2 * I + 1});
      Refs[Reg].push_back(I);
    }

  auto conflicts = [&](MCPhysReg Reg, int B, int F) {
    for (auto [BB, BF] : Busy[Reg])
      if (std::max(B, BB) <= std::min(F, BF))
        return true;
    return false;
  };
  auto lastRef = [&](MCPhysReg Reg, int S) {
    int Last = INT_MIN;
    for (int T : Refs[Reg])
      if (T <= S && T > Last)
        Last = T;
    return Last;
  };

  // Assign renamable ranges in program order.
  std::vector<int> Order;
  for (size_t K = 0; K < Ranges.size(); ++K)
    if (!Ranges[K].Fixed)
      Order.push_back(K);
  std::stable_sort(Order.begin(), Order.end(), [&](int A, int B) {
    return Ranges[A].Start < Ranges[B].Start;
  });

  std::vector<MCPhysReg> Assigned(Ranges.size(), 0);
  for (int K : Order) {
    const Range &R = Ranges[K];
    MCPhysReg Best = 0;
    int BestLast = INT_MAX;
    for (MCPhysReg Reg : *RC[K]) {
      if (!Candidate.test(Reg) && Reg != R.Reg)
        continue;
      if (conflicts(Reg, R.Begin, R.Finish))
        continue;
      int Last = lastRef(Reg, R.Start);
      // Least recently used wins; on a tie keep the original register.
      if (Last < BestLast || (Last == BestLast && Reg == R.Reg)) {
        Best = Reg;
        BestLast = Last;
      }
    }
    if (!Best) {
      // The greedy order painted itself into a corner: keep the block as is.
      ++NumBlocksSkipped;
      return false;
    }
    Assigned[K] = Best;
    addRefs(Best, R);
  }

  // Value replay for the self-check: the register each operand names before
  // renaming.
  DenseMap<const MachineOperand *, MCPhysReg> OrigReg;
  if (MuonHazardRenameVerify)
    for (MachineInstr *MI : Instrs)
      for (const MachineOperand &MO : MI->operands())
        if (MO.isReg() && MO.getReg().isPhysical())
          OrigReg[&MO] = MO.getReg();

  bool Changed = false;
  for (int K : Order) {
    if (Assigned[K] == Ranges[K].Reg)
      continue;
    for (MachineOperand *MO : Ranges[K].Ops)
      MO->setReg(Assigned[K]);
    ++NumRangesRenamed;
    Changed = true;
  }
  if (Changed && MuonHazardRenameVerify)
    verifyBlock(MBB, Instrs, OrigReg, LiveOut);
  return Changed;
}

// Replays the block twice, once with the original and once with the renamed
// registers, giving every write a value number. Every read must see the same
// value in both, and so must every register live out of the block.
void RISCVMuonHazardRename::verifyBlock(
    MachineBasicBlock &MBB, ArrayRef<MachineInstr *> Instrs,
    const DenseMap<const MachineOperand *, MCPhysReg> &OrigReg,
    const LivePhysRegs &LiveOut) {
  // Value numbers: -1 - Reg for a value live into the block, -1000000 - I for
  // a call clobber at instruction I, and 2 * (operand count so far) for writes.
  DenseMap<MCPhysReg, int> Before, After;
  auto valueOf = [](DenseMap<MCPhysReg, int> &V, MCPhysReg Reg) {
    auto It = V.find(Reg);
    return It == V.end() ? -1 - int(Reg) : It->second;
  };
  auto fail = [&](const MachineInstr *MI, const Twine &What) {
    errs() << "muon-hazard-rename: " << What << " in "
           << MBB.getParent()->getName() << " " << printMBBReference(MBB)
           << "\n";
    if (MI)
      MI->print(errs());
    report_fatal_error("muon-hazard-rename self-check failed");
  };
  int Next = 0;
  // Early-clobber results are written before the instruction has read its
  // inputs, so they are applied first.
  auto writeDefs = [&](MachineInstr *MI, bool EarlyClobber) {
    for (const MachineOperand &MO : MI->operands()) {
      if (!MO.isReg() || !MO.isDef() || !MO.getReg() ||
          MO.isEarlyClobber() != EarlyClobber)
        continue;
      MCPhysReg New = MO.getReg(), Old = OrigReg.lookup(&MO);
      if (!isGPR(New) || New == RISCV::X0)
        continue;
      Before[Old] = Next;
      After[New] = Next;
      Next += 2;
    }
  };
  for (int I = 0, N = Instrs.size(); I < N; ++I) {
    MachineInstr *MI = Instrs[I];
    writeDefs(MI, true);
    for (const MachineOperand &MO : MI->operands()) {
      if (!MO.isReg() || !MO.isUse() || MO.isUndef() || !MO.getReg())
        continue;
      MCPhysReg New = MO.getReg(), Old = OrigReg.lookup(&MO);
      if (!isGPR(New) || New == RISCV::X0)
        continue;
      if (valueOf(Before, Old) != valueOf(After, New))
        fail(MI, "read sees a different value");
    }
    for (const MachineOperand &MO : MI->operands())
      if (MO.isRegMask())
        for (MCPhysReg Reg : RISCV::GPRRegClass)
          if (MO.clobbersPhysReg(Reg)) {
            Before[Reg] = -1000000 - I;
            After[Reg] = -1000000 - I;
          }
    writeDefs(MI, false);
  }
  for (MCPhysReg Reg : RISCV::GPRRegClass)
    if (Reg != RISCV::X0 && !LiveOut.available(*MRI, Reg) &&
        valueOf(Before, Reg) != valueOf(After, Reg))
      fail(nullptr, Twine("live-out register x") + Twine(Reg - RISCV::X0) +
                        " holds a different value");
}
