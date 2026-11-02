//==- RISCVSchedMuon.td - Muon SIMT core scheduling model ----*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the simpler per-operand machine model.
// This works with MachineScheduler. See MCSchedule.h for details.

// Muon (Radiance SIMT core) model, derived from the Rocket model. Each warp is
// admitted in order into a small reservation station, so the model is
// in-order; the latencies are the result latencies seen by one warp (shared
// memory loads ~35 cycles, the FP16 pipe ~6, fdiv.h ~24). Used with
// -mtune=muon so that the scheduler separates producers from consumers.
def MuonModel : SchedMachineModel {
  let MicroOpBufferSize = 0;
  let IssueWidth = 1;
  let LoadLatency = 30;
  let MispredictPenalty = 3;
  let CompleteModel = false;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZcmt, HasStdExtZknd, HasStdExtZkne,
                             HasStdExtZknh, HasStdExtZksed, HasStdExtZksh,
                             HasStdExtZkr, HasVInstructions, HasVInstructionsI64];
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Each pipeline is a ProcResource with BufferSize = 0 (in order).

let BufferSize = 0 in {
def MuonUnitALU        : ProcResource<1>; // Int ALU
def MuonUnitIMul       : ProcResource<1>; // Int Multiply
def MuonUnitMem        : ProcResource<1>; // Load/Store
def MuonUnitB          : ProcResource<1>; // Branch

def MuonUnitFPALU      : ProcResource<1>; // FP ALU
}

let BufferSize = 1 in {
def MuonUnitIDiv       : ProcResource<1>; // Int Division
def MuonUnitFPDivSqrt  : ProcResource<1>; // FP Divide/Sqrt
}

//===----------------------------------------------------------------------===//

let SchedModel = MuonModel in {

// Branching
def : WriteRes<WriteJmp, [MuonUnitB]>;
def : WriteRes<WriteJal, [MuonUnitB]>;
def : WriteRes<WriteJalr, [MuonUnitB]>;

// Integer arithmetic and logic
let Latency = 2 in {
def : WriteRes<WriteIALU32, [MuonUnitALU]>;
def : WriteRes<WriteIALU, [MuonUnitALU]>;
def : WriteRes<WriteShiftImm32, [MuonUnitALU]>;
def : WriteRes<WriteShiftImm, [MuonUnitALU]>;
def : WriteRes<WriteShiftReg32, [MuonUnitALU]>;
def : WriteRes<WriteShiftReg, [MuonUnitALU]>;
}

// Integer multiplication
let Latency = 4 in {
def : WriteRes<WriteIMul, [MuonUnitIMul]>;
def : WriteRes<WriteIMul32, [MuonUnitIMul]>;
}

// Integer division
// Worst case latency is used.
def : WriteRes<WriteIDiv32, [MuonUnitIDiv]> {
  let Latency = 34;
  let ReleaseAtCycles = [34];
}
def : WriteRes<WriteIDiv, [MuonUnitIDiv]> {
  let Latency = 33;
  let ReleaseAtCycles = [33];
}

// Memory
def : WriteRes<WriteSTB, [MuonUnitMem]>;
def : WriteRes<WriteSTH, [MuonUnitMem]>;
def : WriteRes<WriteSTW, [MuonUnitMem]>;
def : WriteRes<WriteSTD, [MuonUnitMem]>;
def : WriteRes<WriteFST32, [MuonUnitMem]>;
def : WriteRes<WriteFST64, [MuonUnitMem]>;

let Latency = 30 in {
def : WriteRes<WriteLDB, [MuonUnitMem]>;
def : WriteRes<WriteLDH, [MuonUnitMem]>;
def : WriteRes<WriteLDW, [MuonUnitMem]>;
def : WriteRes<WriteLDD, [MuonUnitMem]>;
def : WriteRes<WriteFLD32, [MuonUnitMem]>;
def : WriteRes<WriteFLD64, [MuonUnitMem]>;

// Atomic memory
def : WriteRes<WriteAtomicW, [MuonUnitMem]>;
def : WriteRes<WriteAtomicD, [MuonUnitMem]>;

def : WriteRes<WriteAtomicLDW, [MuonUnitMem]>;
def : WriteRes<WriteAtomicLDD, [MuonUnitMem]>;
}

def : WriteRes<WriteAtomicSTW, [MuonUnitMem]>;
def : WriteRes<WriteAtomicSTD, [MuonUnitMem]>;

// Single precision.
let Latency = 6 in {
def : WriteRes<WriteFAdd32, [MuonUnitFPALU]>;
def : WriteRes<WriteFSGNJ32, [MuonUnitFPALU]>;
def : WriteRes<WriteFMinMax32, [MuonUnitFPALU]>;
}

// Double precision
let Latency = 6 in {
def : WriteRes<WriteFAdd64, [MuonUnitFPALU]>;
def : WriteRes<WriteFSGNJ64, [MuonUnitFPALU]>;
def : WriteRes<WriteFMinMax64, [MuonUnitFPALU]>;
}

// Conversions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF32, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtI32ToF64, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtI64ToF32, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtI64ToF64, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF32ToI32, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF32ToI64, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF64ToI32, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF64ToI64, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF32ToF64, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF64ToF32, [MuonUnitFPALU]>;

def : WriteRes<WriteFClass32, [MuonUnitFPALU]>;
def : WriteRes<WriteFClass64, [MuonUnitFPALU]>;
def : WriteRes<WriteFCmp32, [MuonUnitFPALU]>;
def : WriteRes<WriteFCmp64, [MuonUnitFPALU]>;
def : WriteRes<WriteFMovF32ToI32, [MuonUnitFPALU]>;
def : WriteRes<WriteFMovI32ToF32, [MuonUnitFPALU]>;
def : WriteRes<WriteFMovF64ToI64, [MuonUnitFPALU]>;
def : WriteRes<WriteFMovI64ToF64, [MuonUnitFPALU]>;
}

// FP multiplication
let Latency = 6 in {
def : WriteRes<WriteFMul32, [MuonUnitFPALU]>;
def : WriteRes<WriteFMA32, [MuonUnitFPALU]>;
}

let Latency = 7 in {
def : WriteRes<WriteFMul64, [MuonUnitFPALU]>;
def : WriteRes<WriteFMA64, [MuonUnitFPALU]>;
}

// FP division
// FP division unit on Rocket is not pipelined, so set resource cycles to latency.
let Latency = 24, ReleaseAtCycles = [20] in {
def : WriteRes<WriteFDiv32, [MuonUnitFPDivSqrt]>;
def : WriteRes<WriteFDiv64, [MuonUnitFPDivSqrt]>;
}

// FP square root unit on Rocket is not pipelined, so set resource cycles to latency.
def : WriteRes<WriteFSqrt32, [MuonUnitFPDivSqrt]> { let Latency = 20;
                                                      let ReleaseAtCycles = [20]; }
def : WriteRes<WriteFSqrt64, [MuonUnitFPDivSqrt]> { let Latency = 25;
                                                      let ReleaseAtCycles = [25]; }

// Half precision (bf16 in the Muon FPU)
let Latency = 6 in {
def : WriteRes<WriteFAdd16, [MuonUnitFPALU]>;
def : WriteRes<WriteFMul16, [MuonUnitFPALU]>;
def : WriteRes<WriteFMA16, [MuonUnitFPALU]>;
def : WriteRes<WriteFSGNJ16, [MuonUnitFPALU]>;
def : WriteRes<WriteFMinMax16, [MuonUnitFPALU]>;
def : WriteRes<WriteFClass16, [MuonUnitFPALU]>;
def : WriteRes<WriteFCmp16, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF16ToI32, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF16ToI64, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF16ToF32, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF16ToF64, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtI32ToF16, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtI64ToF16, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF32ToF16, [MuonUnitFPALU]>;
def : WriteRes<WriteFCvtF64ToF16, [MuonUnitFPALU]>;
def : WriteRes<WriteFMovI16ToF16, [MuonUnitFPALU]>;
def : WriteRes<WriteFMovF16ToI16, [MuonUnitFPALU]>;
}
let Latency = 30 in
def : WriteRes<WriteFLD16, [MuonUnitMem]>;
def : WriteRes<WriteFST16, [MuonUnitMem]>;
let Latency = 24, ReleaseAtCycles = [20] in {
def : WriteRes<WriteFDiv16, [MuonUnitFPDivSqrt]>;
def : WriteRes<WriteFSqrt16, [MuonUnitFPDivSqrt]>;
}
def : ReadAdvance<ReadFAdd16, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFCvtF16ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF16, 0>;
def : ReadAdvance<ReadFCvtI64ToF16, 0>;
def : ReadAdvance<ReadFCvtF32ToF16, 0>;
def : ReadAdvance<ReadFCvtI32ToF16, 0>;
def : ReadAdvance<ReadFCvtF16ToI64, 0>;
def : ReadAdvance<ReadFCvtF16ToF32, 0>;
def : ReadAdvance<ReadFCvtF16ToI32, 0>;
def : ReadAdvance<ReadFDiv16, 0>;
def : ReadAdvance<ReadFCmp16, 0>;
def : ReadAdvance<ReadFMA16, 0>;
def : ReadAdvance<ReadFMA16Addend, 0>;
def : ReadAdvance<ReadFMinMax16, 0>;
def : ReadAdvance<ReadFMul16, 0>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
def : ReadAdvance<ReadFMovF16ToI16, 0>;
def : ReadAdvance<ReadFSGNJ16, 0>;
def : ReadAdvance<ReadFSqrt16, 0>;

// Others
def : WriteRes<WriteCSR, []>;
def : WriteRes<WriteNop, []>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedV;
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedSFB;
}
