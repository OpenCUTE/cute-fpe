//记得修改if相关，改成mux
package cute

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
class RawFloatException extends Bundle {
    val is_nan = Bool()
    val is_inf = Bool()
    val is_zero = Bool()
  }

  class RawFloat(val expWidth: Int, val precision: Int) extends Bundle {
    val sign = Bool()
    val exp = SInt(expWidth.W)
    val signed_sig = SInt((precision + 1).W)
    val exception = new RawFloatException
  }

  object RawFloat {
    // 不能用于e2m1和e4m3！这两个的异常值定义不同，e2m1没有异常值，e4m3只在0bx1111111时为NAN，指数全一为最大指数，但在这里会被处理成负数
    def fromUInt(x: UInt, expWidth: Int, pc: Int, E4M3: Boolean = false, E2M1: Boolean = false): RawFloat = {
      val fp = Wire(new RawFloat(expWidth, pc))
      val nz =  x(expWidth + pc - 2, pc - 1).orR
      val sign = x(expWidth + pc - 1)
      fp.sign := sign
      val bias = Wire(UInt(expWidth.W))
      bias := (1.U << (expWidth - 1)) - 1.U //
      fp.exp := ((x(expWidth + pc - 2, pc - 1) | !nz).asUInt - bias).asSInt
      val sig = Cat(0.U(1.W), nz, x(pc - 2, 0))
    //   fp.signed_sig := Mux(sign, -sig.asSInt.pad(pc + 1), sig.asSInt.pad(pc + 1))
      fp.signed_sig := sig.asSInt
      // 不能用e4m3
    //   if (E4M3) {
    //     fp.exception.is_nan := x(expWidth + pc - 2, 0).andR 
    //     fp.exception.is_inf := false.B
    //   }
    //   else 
      if (E2M1) {
        fp.exception.is_nan := false.B
        fp.exception.is_inf := false.B
      }
      else {
        fp.exception.is_nan := x(expWidth + pc - 2, pc - 1).andR && x(pc - 2, 0).orR
        fp.exception.is_inf := x(expWidth + pc - 2, pc - 1).andR && !x(pc - 2, 0).orR
      }
      fp.exception.is_zero := !x(expWidth + pc - 2, 0).orR
      fp
    }

    // e4m3单独处理，在计算时用5位表示exp
    def frome4m3(x: UInt): RawFloat = {
        val fp = Wire(new RawFloat(5, 4))
        val nz =  x(4 + 4 - 2, 4 - 1).orR
        val sign = x(4 + 4 - 1)
        fp.sign := sign
        val bias = Wire(UInt(5.W))
        bias := (1.U << (4 - 1)) - 1.U //
        fp.exp := ((x(4 + 4 - 2, 4 - 1) | !nz).asUInt.pad(5) - bias).asSInt
        val sig = Cat(0.U(1.W), nz, x(4 - 2, 0))
        //   fp.signed_sig := Mux(sign, -sig.asSInt.pad(pc + 1), sig.asSInt.pad(pc + 1))
        fp.signed_sig := sig.asSInt

        fp.exception.is_nan := x(4 + 4 - 2, 0).andR 
        fp.exception.is_inf := false.B
        fp.exception.is_zero := !x(4 + 4 - 2, 0).orR
        fp
    }
}

class FP4toint(implicit p: Parameters) extends CuteModule {   // 输出格式为xxxx.x，00001为0.5
    val io = IO(new Bundle{
        val in = Input(UInt(4.W))
        val out = Output(SInt(5.W))
    })

    val sign = io.in(3)
    val exp = io.in(2, 1)
    val mantissa = io.in(0)
    val nz = exp.orR
    val sig = Cat(0.U(3.W), nz, mantissa).asSInt // 5位无符号数
    val unsignedSig = Wire(SInt(5.W))

    when(exp === 2.U){
        unsignedSig := sig << 1.U
    }.elsewhen(exp === 3.U){
        unsignedSig := sig << 2.U
    }.otherwise{
        unsignedSig := sig
    }

    io.out := Mux(sign, -unsignedSig.asSInt, unsignedSig.asSInt)
}

class CLZ(len: Int)(implicit p: Parameters) extends CuteModule {

  val inWidth = len
  val outWidth = (inWidth - 1).U.getWidth

  val io = IO(new Bundle() {
    val in = Input(UInt(inWidth.W))
    val out = Output(UInt(outWidth.W))
  })

  io.out := PriorityEncoder(io.in.asBools.reverse)
}

class FDecodeResult(implicit p: Parameters) extends CuteBundle{
    // val Int8Vec = Vec(ReduceWidth/8, UInt(9.W))
    val TF32Vec = Vec(ReduceWidth/8, new RawFloat(10, 11))
}

class FPScaleDecoder(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(Vec(ReduceWidth/4/MinGroupSize, UInt(8.W)))
        val out = Output(Vec(ReduceWidth/4/MinGroupSize, new RawFloat(8, 11)))
    })

    for(i <- 0 until ReduceWidth/4/MinGroupSize){
        val Bits8  = io.in(i)
        val DecodeE4M3 = RawFloat.frome4m3(Bits8)
        val E4M3toTF32 = Wire(new RawFloat(8, 11))
        E4M3toTF32.exp := DecodeE4M3.exp.pad(8)
        E4M3toTF32.signed_sig := DecodeE4M3.signed_sig
        E4M3toTF32.exception := DecodeE4M3.exception
        E4M3toTF32.sign := DecodeE4M3.sign
        io.out(i) := E4M3toTF32
    }
}

class FVecDecoder(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(UInt(ReduceWidth.W))
        val opcode = Input(UInt(4.W))   //0:Int8, 1:FP16, 2:BF16, 3:TF32, 4:UI8， 7,11：E4M3, 8,12:e5m2
        val out = Output(new FDecodeResult)
    })

    val TF32Zero = RawFloat.fromUInt(0.U(21.W), 10, 11)

    val TF32i8 = Wire(Vec(ReduceWidth/8, new RawFloat(10, 11)))
    for(i <- 0 until ReduceWidth/8){
        TF32i8(i) := RawFloat.fromUInt(0.U(21.W), 10, 11)
        TF32i8(i).signed_sig := Mux(io.opcode === 0.U, 
            io.in(8 * i + 7, 8 * i).asSInt.pad(12), io.in(8 * i + 7, 8 * i).pad(12).asSInt
        )
    }

    for(i <- 0 until ReduceWidth/32){
        val Bits8  = io.in(8 * i + 7, 8 * i)
        val Bits16 = io.in(16 * i + 15, 16 * i)
        val Bits32 = io.in(32 * i + 31, 32 * i)
        val DecodeFP16 = RawFloat.fromUInt(Bits16, 5, 11)
        val DecodeBF16 = RawFloat.fromUInt(Bits16, 8, 8)
        val DecodeE4M3 = RawFloat.frome4m3(Bits8)
        val DecodeE5M2 = RawFloat.fromUInt(Bits8, 5, 3)
        val DecodeTF32 = RawFloat.fromUInt(Bits32(31, 13), 8, 11)
        val FP16toTF32 = Wire(new RawFloat(10, 11))
        val BF16toTF32 = Wire(new RawFloat(10, 11))
        val E4M3toTF32 = Wire(new RawFloat(10, 11))
        val E5M2toTF32 = Wire(new RawFloat(10, 11))
        val TF32conv = Wire(new RawFloat(10, 11))
        FP16toTF32.exp := DecodeFP16.exp.pad(10)
        FP16toTF32.signed_sig := DecodeFP16.signed_sig
        FP16toTF32.exception := DecodeFP16.exception
        FP16toTF32.sign := DecodeFP16.sign
        BF16toTF32.exp := DecodeBF16.exp
        BF16toTF32.signed_sig := Cat(DecodeBF16.signed_sig, 0.U(3.W)).asSInt
        BF16toTF32.exception := DecodeBF16.exception
        BF16toTF32.sign := DecodeBF16.sign
        E4M3toTF32.exp := DecodeE4M3.exp.pad(10)
        E4M3toTF32.signed_sig := Cat(DecodeE4M3.signed_sig, 0.U(7.W)).asSInt
        E4M3toTF32.exception := DecodeE4M3.exception
        E4M3toTF32.sign := DecodeE4M3.sign
        E5M2toTF32.exp := DecodeE5M2.exp.pad(10)
        E5M2toTF32.signed_sig := Cat(DecodeE5M2.signed_sig, 0.U(8.W)).asSInt
        E5M2toTF32.exception := DecodeE5M2.exception
        E5M2toTF32.sign := DecodeE5M2.sign
        TF32conv.exp := DecodeTF32.exp.pad(10)
        TF32conv.signed_sig := DecodeTF32.signed_sig
        TF32conv.exception := DecodeTF32.exception
        TF32conv.sign := DecodeTF32.sign
        io.out.TF32Vec(i) := Mux(
            io.opcode === 1.U, FP16toTF32,
            Mux(io.opcode === 2.U, BF16toTF32,
                Mux(io.opcode === 3.U, TF32conv, 
                    Mux(io.opcode === 7.U || io.opcode === 11.U, E4M3toTF32,
                        Mux(io.opcode === 8.U || io.opcode === 12.U, E5M2toTF32,
                            TF32i8(i)
                        )
                    )
                )
            )
        )
    }

    for(i <- ReduceWidth/32 until ReduceWidth/16){
        val Bits8  = io.in(8 * i + 7, 8 * i)
        val Bits16 = io.in(16 * i + 15, 16 * i)
        val DecodeFP16 = RawFloat.fromUInt(Bits16, 5, 11)
        val DecodeBF16 = RawFloat.fromUInt(Bits16, 8, 8)
        val DecodeE4M3 = RawFloat.frome4m3(Bits8)
        val DecodeE5M2 = RawFloat.fromUInt(Bits8, 5, 3)
        val FP16toTF32 = Wire(new RawFloat(10, 11))
        val BF16toTF32 = Wire(new RawFloat(10, 11))
        val E4M3toTF32 = Wire(new RawFloat(10, 11))
        val E5M2toTF32 = Wire(new RawFloat(10, 11))
        FP16toTF32.exp := DecodeFP16.exp.pad(10)
        FP16toTF32.signed_sig := DecodeFP16.signed_sig
        FP16toTF32.exception := DecodeFP16.exception
        FP16toTF32.sign := DecodeFP16.sign
        BF16toTF32.exp := DecodeBF16.exp
        BF16toTF32.signed_sig := Cat(DecodeBF16.signed_sig, 0.U(3.W)).asSInt// 有符号数，所以12位
        BF16toTF32.exception := DecodeBF16.exception
        BF16toTF32.sign := DecodeBF16.sign
        E4M3toTF32.exp := DecodeE4M3.exp.pad(10)
        E4M3toTF32.signed_sig := Cat(DecodeE4M3.signed_sig, 0.U(7.W)).asSInt
        E4M3toTF32.exception := DecodeE4M3.exception
        E4M3toTF32.sign := DecodeE4M3.sign
        E5M2toTF32.exp := DecodeE5M2.exp.pad(10)
        E5M2toTF32.signed_sig := Cat(DecodeE5M2.signed_sig, 0.U(8.W)).asSInt
        E5M2toTF32.exception := DecodeE5M2.exception
        E5M2toTF32.sign := DecodeE5M2.sign
        io.out.TF32Vec(i) := Mux(
            io.opcode === 1.U, FP16toTF32,
            Mux(io.opcode === 2.U, BF16toTF32, 
                Mux(io.opcode === 3.U, TF32Zero, 
                    Mux(io.opcode === 7.U || io.opcode === 11.U, E4M3toTF32,
                        Mux(io.opcode === 8.U || io.opcode === 12.U, E5M2toTF32,
                            TF32i8(i)
                        )
                    )
                )
            )
        )
    }

    for(i <- ReduceWidth/16 until ReduceWidth/8){
        val Bits8  = io.in(8 * i + 7, 8 * i)
        val DecodeE4M3 = RawFloat.frome4m3(Bits8)
        val DecodeE5M2 = RawFloat.fromUInt(Bits8, 5, 3)
        val E4M3toTF32 = Wire(new RawFloat(10, 11))
        val E5M2toTF32 = Wire(new RawFloat(10, 11))
        E4M3toTF32.exp := DecodeE4M3.exp.pad(10)
        E4M3toTF32.signed_sig := Cat(DecodeE4M3.signed_sig, 0.U(7.W)).asSInt
        E4M3toTF32.exception := DecodeE4M3.exception
        E4M3toTF32.sign := DecodeE4M3.sign
        E5M2toTF32.exp := DecodeE5M2.exp.pad(10)
        E5M2toTF32.signed_sig := Cat(DecodeE5M2.signed_sig, 0.U(8.W)).asSInt
        E5M2toTF32.exception := DecodeE5M2.exception
        E5M2toTF32.sign := DecodeE5M2.sign
        io.out.TF32Vec(i) := Mux(
            io.opcode === 7.U || io.opcode === 11.U, E4M3toTF32,
            Mux(io.opcode === 8.U || io.opcode === 12.U, E5M2toTF32,
                TF32i8(i)
            )
        )
    }
}

class CmpTreeP0Res(layers:Int, elemNum:Int, expWidth:Int)(implicit p: Parameters) extends CuteBundle{
    val part1   = UInt((elemNum).W)   //第6层mask（也就是mask(5)）,用于算后续的mask
    val part2   = Vec(expWidth - layers, UInt((elemNum).W)) //6层以后的数据传到下一层
    val part3   = UInt(layers.W) //前6层的结果
}

class CmpTreeP0(layers:Int, elemNum:Int, expWidth:Int)(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle {
        val in    = Input(Vec(elemNum, UInt(expWidth.W)))
        val out   = Output(new CmpTreeP0Res(layers, elemNum, expWidth))
    })

  // 生成 in_vec_0: 9组，每组ReduceWidth/16位
    val inVec0 = Wire(Vec(expWidth, UInt((elemNum).W)))
    for (j <- 0 until expWidth) {
        inVec0(j) := Cat((0 until elemNum).reverse.map(i => io.in(i)(expWidth - 1 - j)))
    }

  // 生成 mask: 9组，每组ReduceWidth/16位
    val mask = Wire(Vec(layers, UInt((elemNum).W)))
    mask(0) := Mux(inVec0(0) === 0.U, ~inVec0(0), inVec0(0))
    for (i <- 1 until layers) {
        val prev = Wire(UInt((elemNum).W))
        val curr = Wire(UInt((elemNum).W))
        val prevandcurr = Wire(UInt((elemNum).W))
        prev := mask(i - 1)
        curr := inVec0(i)
        prevandcurr := (prev & curr)
        mask(i) := Mux(prevandcurr === 0.U, prev, prevandcurr)
    }

  // 生成 out: 9位
    io.out.part1 := mask(layers - 1)
    for(i <- 0 until (expWidth - layers)){
        io.out.part2(i) := inVec0(i + layers)
    }

    val res = Wire(Vec(layers, UInt(1.W)))
    for(i <- 0 until layers){
        val bit = Wire(UInt((elemNum).W))
        bit := mask(layers - 1) & inVec0(i)
        res(i) := Mux(bit === 0.U, 0.U, 1.U)
    }

    io.out.part3 := Cat((0 until layers).map(k => res(k)))
}

class CmpTreeP1(layers : Int, elemNum:Int, expWidth:Int)(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle {
        val in    = Input(new CmpTreeP0Res(layers, elemNum, expWidth))
        val out   = Output(UInt(expWidth.W))
    })

    val mask = Wire(Vec(expWidth - layers, UInt((elemNum).W)))
    for (i <- 0 until (expWidth - layers)) {
        val prev = Wire(UInt((elemNum).W))
        val curr = Wire(UInt((elemNum).W))
        val prevandcurr = Wire(UInt((elemNum).W))
        if(i != 0){
            prev := mask(i - 1)
        }else{
            prev := io.in.part1
        }
        curr := io.in.part2(i)
        prevandcurr := (prev & curr)
        mask(i) := Mux(prevandcurr === 0.U, prev, prevandcurr)
    }

    val res = Wire(Vec(expWidth - layers, UInt(1.W)))
    for(i <- 0 until (expWidth - layers)){
        val bit = Wire(UInt(elemNum.W))
        bit := mask(expWidth - 1 - layers) & io.in.part2(i)
        res(i) := Mux(bit === 0.U, 0.U, 1.U)
    }

    val partres = Wire(UInt((expWidth - layers).W))
    partres := Cat((0 until (expWidth - layers)).map(k => res(k)))

  // 生成 out: 9位
    io.out := Cat(io.in.part3, partres)
}

class FPipe0Result(implicit p: Parameters) extends CuteBundle{
    val Product0 = Vec(ReduceWidth/16, SInt(23.W))
    val Product1 = Vec(ReduceWidth/16, SInt(23.W))
    val FP4ReduceRes = Vec(ReduceWidth/4/FP4P0AddNum, SInt((10 + log2Ceil(FP4P0AddNum)).W))
    val CMantissa = SInt(32.W)
    val ExceptionVec = Vec(ReduceWidth/8 + 1, new RawFloatException) //每个向量的异常标志位
    val MulExpVec = Vec(ReduceWidth/8 + 1, UInt(10.W))
    // val CmpTreeP0Result = new CmpTreeP0Res(cmptreelayers, ReduceWidth/16 + 1, 9)
    val CmpTreefp8P0Result = new CmpTreeP0Res(fp8cmptreelayers, ReduceWidth/8 + 1, 10)
    val SignVec = Vec(ReduceWidth/8 + 1, Bool()) //每个向量的符号位
    val opcode = UInt(4.W)   //0:Int8, 1:FP16, 2:BF16, 3:TF32
    val scaleFP = Vec(ReduceWidth/4/MinGroupSize + 1, new RawFloat(9, 11))
}

class FPipe1Result(implicit p: Parameters) extends CuteBundle{
    val Product0 = Vec(ReduceWidth/16, SInt(23.W))
    val Product1 = Vec(ReduceWidth/16, SInt(23.W))
    val CMantissa = SInt(32.W)
    val RightShiftVec = Vec((ReduceWidth/8) + 1, UInt(10.W))
    val FP4ReduceRes = Vec(ReduceWidth/4/16, SInt((10 + log2Ceil(16)).W))
    val MaxExp = UInt(10.W)
    val CmpTreefp4P0Result = new CmpTreeP0Res(8, ReduceWidth/4/MinGroupSize + 1, 9)
    // val FP4MaxExp = SInt(8.W)
    val opcode = UInt(4.W)  //0:Int8, 1:FP16, 2:BF16, 3:TF32
    val SumException = UInt(4.W) //归约计算结果的异常标志位
    val SignVec = Vec(ReduceWidth/8 + 1, Bool()) //每个向量的符号位
    val scaleFP = Vec(ReduceWidth/4/MinGroupSize + 1, new RawFloat(9, 11))
}

class FPipe2Result(implicit p: Parameters) extends CuteBundle{
    val ReduceRes0 = Vec(P3AddNum, SInt((26 + log2Ceil(P2AddNum)).W))
    val ReduceRes1 = Vec(P3AddNum, SInt((26 + log2Ceil(P2AddNum)).W)) // ReduceRes0是尾数右移后的结果，ReduceRes1是Int8的结果
    val CMantissa = SInt(32.W) // C的尾数
    val MaxExp = UInt(10.W)
    val opcode = UInt(4.W)  //0:Int8, 1:FP16, 2:BF16, 3:TF32
    val SumException = UInt(4.W) //归约计算结果的异常标志位
    val FP4ABShift = Vec(ReduceWidth/4/MinGroupSize + 1, SInt((32 + 3).W))
}

class FPipe3Result(implicit p: Parameters) extends CuteBundle{
    // val ReduceRes = SInt(32.W) //归约计算结果
    val ReduceRes0 = SInt((32).W)
    val ReduceRes1 = SInt((32).W) // ReduceRes0是尾数右移后的结果，ReduceRes1是Int8的结果
    // val ReduceResFP4 = SInt(32.W)
    val CMantissa = SInt(32.W) // C的尾数
    val MaxExp = UInt(10.W)
    val opcode = UInt(4.W)  //0:Int8, 1:FP16, 2:BF16, 3:TF32
    val SumException = UInt(4.W) //归约计算结果的异常标志位
}

// Pipe0的功能是并行完成{计算尾数乘积}和{求阶码最大值，计算右移位数}
class FReduceMACPipe0(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val inA = Input(new FDecodeResult)
        val inB = Input(new FDecodeResult)
        val inAFP4Vec = Input(Vec(ReduceWidth/4, SInt(5.W)))
        val inBFP4Vec = Input(Vec(ReduceWidth/4, SInt(5.W)))
        val inAscale = Input(Vec(ReduceWidth/4/MinGroupSize, UInt(8.W)))
        val inBscale = Input(Vec(ReduceWidth/4/MinGroupSize, UInt(8.W)))
        val inC = Input(UInt(32.W))
        val opcode = Input(UInt(4.W))   //0:Int8, 1:FP16, 2:BF16, 3:TF32, 4:I8 * UI8, 5:UI8 * I8, 6:UI8 * UI8, 7:MXFP8E4M3, 8:MXFP8E5M2, 9:NVFP4, 10:MXFP4, 11:E4M3, 12:E5M2
        val out = Output(new FPipe0Result)
    })

    val CDecode = RawFloat.fromUInt(io.inC, 8, 24)

    // FP4 scale 相乘
    val FP4AScaleDecoder = Module(new FPScaleDecoder())
    val FP4BScaleDecoder = Module(new FPScaleDecoder())
    FP4AScaleDecoder.io.in := io.inAscale
    FP4BScaleDecoder.io.in := io.inBscale

    val mxfp4ScaleSum = Wire(Vec(ReduceWidth/4/32, UInt(10.W)))
    val mxfp4ScaleSat = Wire(Vec(ReduceWidth/4/32, UInt(9.W)))
    // 结果指数超过127即为无穷大，因此不需要保留过大的scale，直接截断到9位，超过9位的和接近9位的结果都会是无穷大
    for (i <- 0 until ReduceWidth/4/32){
        mxfp4ScaleSum(i) := io.inAscale(i).asUInt.pad(10) + io.inBscale(i).asUInt.pad(10) + (1 + 6).U
        mxfp4ScaleSat(i) := Mux(mxfp4ScaleSum(i)(9) === 1.U, 511.U, mxfp4ScaleSum(i)(8, 0))
    }

    val FP4AScaleExceptionVec = Wire(Vec(ReduceWidth/4/MinGroupSize, new RawFloatException))
    val FP4BScaleExceptionVec = Wire(Vec(ReduceWidth/4/MinGroupSize, new RawFloatException))
    val mxfp4AScaleExceptionVec = Wire(Vec(ReduceWidth/4/32, new RawFloatException))
    val mxfp4BScaleExceptionVec = Wire(Vec(ReduceWidth/4/32, new RawFloatException))
    mxfp4AScaleExceptionVec := 0.U.asTypeOf(Vec(ReduceWidth/4/32, new RawFloatException))
    mxfp4BScaleExceptionVec := 0.U.asTypeOf(Vec(ReduceWidth/4/32, new RawFloatException))
    // mxfp4的scale为e8m0，255表示nan
    for (i <- 0 until ReduceWidth/4/32){
        mxfp4AScaleExceptionVec(i).is_nan := io.inAscale(i) === 255.U
        mxfp4BScaleExceptionVec(i).is_nan := io.inBscale(i) === 255.U
    }
    // mxfp4block数量为nvfp4一半
    for (i <- 0 until ReduceWidth/4/MinGroupSize){
        FP4AScaleExceptionVec(i) := Mux(io.opcode === 10.U, mxfp4AScaleExceptionVec(i / 2), FP4AScaleDecoder.io.out(i).exception)
        FP4BScaleExceptionVec(i) := Mux(io.opcode === 10.U, mxfp4BScaleExceptionVec(i / 2), FP4BScaleDecoder.io.out(i).exception)
    }

    io.out.scaleFP(ReduceWidth/4/MinGroupSize).sign := CDecode.sign
    io.out.scaleFP(ReduceWidth/4/MinGroupSize).exp := CDecode.exp.pad(9) + 255.S
    if (DEBUG_FP4) {
    printf("CDecode.exp:%x\n", CDecode.exp)
    printf("io.out.scaleFP(ReduceWidth/4/MinGroupSize).exp:%x\n", io.out.scaleFP(ReduceWidth/4/MinGroupSize).exp)
    }
    io.out.scaleFP(ReduceWidth/4/MinGroupSize).signed_sig := CDecode.signed_sig
    io.out.scaleFP(ReduceWidth/4/MinGroupSize).exception := CDecode.exception

    for (i <- 0 until ReduceWidth/4/MinGroupSize){
        if (DEBUG_FP4) {
            printf("io.inAscale[%d]: %x\n", i.U, io.inAscale(i))
            printf("io.inBscale[%d]: %x\n", i.U, io.inBscale(i))
            printf("FP4AScaleDecoder.io.out[%d].sign: %x\n", i.U, FP4AScaleDecoder.io.out(i).sign)
            printf("FP4BScaleDecoder.io.out[%d].sign: %x\n", i.U, FP4BScaleDecoder.io.out(i).sign)
            printf("FP4AScaleDecoder.io.out[%d].exp: %x\n", i.U, FP4AScaleDecoder.io.out(i).exp)
            printf("FP4BScaleDecoder.io.out[%d].exp: %x\n", i.U, FP4BScaleDecoder.io.out(i).exp)
        }
        io.out.scaleFP(i).sign := Mux(
            io.opcode === 9.U, 
            FP4AScaleDecoder.io.out(i).sign ^ FP4BScaleDecoder.io.out(i).sign,
            0.B
        )
        io.out.scaleFP(i).exp := Mux(
            io.out.ExceptionVec(i).is_zero || (io.opcode =/= 9.U && io.opcode =/= 10.U), 
             0.S, /*这里的9对应于之后对齐前的左移6，两者合起来15与FP32的尾数对齐，之所以避让出9位空位是因为FP4直接转换成整数计算中间结果位宽很大*/
            Mux(io.opcode === 10.U, mxfp4ScaleSat(i / 2).asSInt, FP4AScaleDecoder.io.out(i).exp.pad(9) + FP4BScaleDecoder.io.out(i).exp.pad(9) + (255 + 9).S)
        )

        if (DEBUG_FP8 || DEBUG_FP4) {
            printf("io.out.scaleFP(%d).sign:%x\n", i.U,io.out.scaleFP(i).sign)
            printf("io.out.scaleFP(%d).exp:%x\n", i.U,io.out.scaleFP(i).exp)
        }

        val sig_mul = Wire(SInt(10.W))
        sig_mul := Mux(
            io.opcode === 9.U, 
            (FP4AScaleDecoder.io.out(i).signed_sig(4, 0).asSInt * FP4BScaleDecoder.io.out(i).signed_sig(4, 0).asSInt).asSInt,
            Mux(io.opcode === 10.U, 1.S, 0.S)
        )
        io.out.scaleFP(i).signed_sig := Mux(io.out.scaleFP(i).sign, 
            -sig_mul.asSInt, sig_mul.asSInt
        ) 
        if (DEBUG_FP4) {
            printf("io.out.scaleFP[%d].sign: %x\n", i.U, io.out.scaleFP(i).sign)
            printf("sig_mul[%d]: %x\n", i.U, sig_mul)
            printf("io.out.scaleFP[%d].signed_sig: %x\n", i.U, io.out.scaleFP(i).signed_sig)
        }

        io.out.scaleFP(i).exception := 0.U.asTypeOf(new RawFloatException)
    }

    //尾数乘法计算
    if (DEBUG_FP8)
    {for (i <- 0 until ReduceWidth/8) {
        printf("inAexp[%d]: %x\n", i.U,io.inA.TF32Vec(i).exp)
        printf("inBexp[%d]: %x\n", i.U,io.inB.TF32Vec(i).exp)
    }}

    // Product0
    // 乘法器位宽要求能够满足12位有符号整数的乘法
    for(i <- 0 until ReduceWidth/16){
        val Product0A = Wire(SInt(12.W))
        val Product0B = Wire(SInt(12.W))

        Product0A := io.inA.TF32Vec(i).signed_sig
        Product0B := io.inB.TF32Vec(i).signed_sig

        io.out.Product0(i) := Product0A * Product0B
        if (DEBUG_FP8)
        {printf("Product0A[%d]: %x\n", i.U,Product0A)
        printf("Product0B[%d]: %x\n", i.U,Product0B)
        printf("product0[%d]: %x\n", i.U,io.out.Product0(i))}
    }

    // Product1
    // 乘法器位宽要求能够满足9位有符号整数的乘法
    for(i <- ReduceWidth/16 until ReduceWidth/8){
        val Product1A = Wire(SInt(12.W))
        val Product1B = Wire(SInt(12.W))

        Product1A := io.inA.TF32Vec(i).signed_sig.asSInt
        Product1B := io.inB.TF32Vec(i).signed_sig.asSInt

        io.out.Product1(i - ReduceWidth/16) := Product1A * Product1B
    }

    val FP4Product = Wire(Vec(ReduceWidth/4, SInt(10.W)))
    // FP4Product
    for (i <- 0 until ReduceWidth/4){
        val FP4A = Wire(SInt(5.W))
        val FP4B = Wire(SInt(5.W))

        FP4A := io.inAFP4Vec(i)
        FP4B := io.inBFP4Vec(i)

        FP4Product(i) := FP4A * FP4B
    }

    
    for(i <- 0 until ReduceWidth/4/FP4P0AddNum){ 
        io.out.FP4ReduceRes(i) := FP4Product.slice(i * FP4P0AddNum, (i + 1) * FP4P0AddNum).reduce(_ + _)
    }

    //阶码计算
    val MulExpVec = Wire(Vec((ReduceWidth/8) + 1, UInt(10.W)))
    val MulExpVecSigned = Wire(Vec((ReduceWidth/8) + 1, SInt(9.W))) //for debug
         
    for(i <- 0 until ReduceWidth/8){

        // printf("inA exp[%d]: %x\n", i.U,io.inA.TF32Vec(i).exp(5, 0))
        // printf("inB exp[%d]: %x\n", i.U,io.inB.TF32Vec(i).exp(5, 0))
        // printf("fp8MulExpVec[%d]: %x\n", i.U,MulExpVec(i))
        MulExpVecSigned(i) := io.inA.TF32Vec(i).exp.pad(9) + io.inB.TF32Vec(i).exp.pad(9)
        MulExpVec(i) := Mux(io.inA.TF32Vec(i).exception.is_zero | io.inB.TF32Vec(i).exception.is_zero, 0.U, 
            (io.inA.TF32Vec(i).exp + io.inB.TF32Vec(i).exp).asUInt + 511.U(10.W))
        if (DEBUG_FP8)
        {printf("MulExpVec[%d]: %x\n", i.U,MulExpVec(i))}
    }

    for(i <- 0 until ReduceWidth/8){
        val AException = Wire(new RawFloatException)
        val BException = Wire(new RawFloatException)
        AException := io.inA.TF32Vec(i).exception
        BException := io.inB.TF32Vec(i).exception
        if (i < ReduceWidth/4/MinGroupSize) {
            AException := Mux(io.opcode === 9.U || io.opcode === 10.U, FP4AScaleExceptionVec(i), io.inA.TF32Vec(i).exception)
            BException := Mux(io.opcode === 9.U || io.opcode === 10.U, FP4BScaleExceptionVec(i), io.inB.TF32Vec(i).exception)
        }

        io.out.ExceptionVec(i).is_nan := AException.is_nan || BException.is_nan ||
            AException.is_inf && BException.is_zero ||
            BException.is_inf && AException.is_zero

        io.out.ExceptionVec(i).is_zero := AException.is_zero && !BException.is_inf ||
            BException.is_zero && !AException.is_inf

        io.out.ExceptionVec(i).is_inf := !AException.is_zero && BException.is_inf ||
            !BException.is_zero && AException.is_inf

        io.out.SignVec(i) := io.inA.TF32Vec(i).sign ^ io.inB.TF32Vec(i).sign
    }

    // printf("io.inC: %x\n", io.inC)
    io.out.SignVec(ReduceWidth/8) := CDecode.sign
    MulExpVec(ReduceWidth/8) := CDecode.exp.pad(10).asUInt + 511.U(10.W) // C的阶码加上偏移量
    if (DEBUG_FP8)
    {printf("io.inC: %x\n", io.inC)
    printf("CDecode exp: %x\n", CDecode.exp.pad(10))
    printf("C exp: %x\n", MulExpVec(ReduceWidth/8))
    }
    MulExpVecSigned(ReduceWidth/8) := CDecode.exp.pad(9) //for debug

    io.out.ExceptionVec(ReduceWidth/8) := CDecode.exception
    io.out.CMantissa := Mux(io.opcode === 1.U || io.opcode === 2.U || io.opcode === 3.U || io.opcode === 7.U || io.opcode === 8.U || io.opcode === 9.U || io.opcode === 10.U || io.opcode === 11.U || io.opcode === 12.U, 
        CDecode.signed_sig.pad(32), io.inC.asSInt)
    
    if (DEBUG_FP8) 
    {printf("c mantissa: %x\n", io.out.CMantissa)}

    //取阶码最大值
    val cmptreefp8p0 = Module(new CmpTreeP0(fp8cmptreelayers, ReduceWidth/8 + 1, 10))
    
    // cmptreep0.io.in(ReduceWidth/16) := MulExpVec(ReduceWidth/8) //最后一个C的阶码
    for (i <- 0 until ReduceWidth/16){
        cmptreefp8p0.io.in(i) := MulExpVec(i)
        cmptreefp8p0.io.in(i + ReduceWidth/16) := Mux(io.opcode === 7.U || io.opcode === 8.U || io.opcode === 11.U || io.opcode === 12.U, MulExpVec(i + ReduceWidth/16), 0.U)
    }

    cmptreefp8p0.io.in(ReduceWidth/8) := MulExpVec(ReduceWidth/8) //最后一个C的阶码
    
    //整理pipe0输出

    //每个乘积的阶码与最大阶码的差值就是右移的位数
    // val RightShiftVec = Wire(Vec(ReduceWidth/16, UInt(9.W)))
    for(i <- 0 until ReduceWidth/8 + 1){
        io.out.MulExpVec(i) := MulExpVec(i)
    }

    //输出opcode和最大阶码
    io.out.opcode := io.opcode
    // io.out.CmpTreeP0Result := cmptreep0.io.out
    io.out.CmpTreefp8P0Result := cmptreefp8p0.io.out
}

// Pipe1的功能是将Pipe0的乘积向量尾数右移，得到尾数向量用于归约计算
class FReduceMACPipe1(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe0Result)
        val out = Output(new FPipe1Result)
    })

    val cmptreefp4p0 = Module(new CmpTreeP0(8, ReduceWidth/4/MinGroupSize + 1, 9))
    // 找到FP4情况下scale product 与C EXP的最大值
    // val FP4MaxExp = Wire(SInt(8.W))
    val ExpVec = Wire(Vec(ReduceWidth/4/MinGroupSize + 1, SInt(9.W)))
    for (i <- 0 until ReduceWidth/4/MinGroupSize + 1){
        ExpVec(i) := io.in.scaleFP(i).exp
        if (DEBUG_FP4) {
            printf("scaleFP[%d].exp: %x\n", i.U, io.in.scaleFP(i).exp)
        }
    }
    // ExpVec(ReduceWidth/4/MinGroupSize) := io.in.MulExpVec(ReduceWidth/8).pad(8).asSInt
    cmptreefp4p0.io.in := ExpVec.map(x => x.asUInt)
    io.out.CmpTreefp4P0Result := cmptreefp4p0.io.out

    // printf("io.out.FP4MaxExp:%x\n", io.out.FP4MaxExp)

    // val cmptreep1 = Module(new CmpTreeP1(cmptreelayers, ReduceWidth/16 + 1, 9))
    // cmptreep1.io.in := io.in.CmpTreeP0Result

    val cmptreefp8p1 = Module(new CmpTreeP1(fp8cmptreelayers, ReduceWidth/8 + 1, 10))
    cmptreefp8p1.io.in := io.in.CmpTreefp8P0Result

    if (DEBUG_FP8)
    {printf("cmptreefp8p1.io.out: %x\n", cmptreefp8p1.io.out)}

    val productMaxExp = Wire(UInt(10.W))
    // productMaxExp := Mux(io.in.opcode === 7.U, cmptreefp8p1.io.out.pad(9), cmptreep1.io.out)
    productMaxExp := cmptreefp8p1.io.out

    // val RightShiftVec = Wire(Vec((ReduceWidth/8) + 1, UInt(10.W)))
    for(i <- 0 until ReduceWidth/8){
        io.out.RightShiftVec(i) := productMaxExp - io.in.MulExpVec(i)
        if (DEBUG_FP8)
        {printf("RightShiftVec[%d]: %x\n", i.U, io.out.RightShiftVec(i) )}
    }
    io.out.RightShiftVec(ReduceWidth/8) := productMaxExp - io.in.MulExpVec(ReduceWidth/8)

    // 处理exception，下一级流水可以根据HasNaN、HasPInf、HasNInf、OnlyNZero来判断结果
    val NaNVec = Wire(UInt((ReduceWidth/8 + 1).W))
    val PInfVec = Wire(UInt((ReduceWidth/8 + 1).W))
    val NInfVec = Wire(UInt((ReduceWidth/8 + 1).W))
    val NZeroVec = Wire(UInt((ReduceWidth/8 + 1).W))

    NaNVec := Cat((0 until (ReduceWidth/8 + 1)).map(i => io.in.ExceptionVec(i).is_nan))
    PInfVec := Cat((0 until (ReduceWidth/8 + 1)).map(i => io.in.ExceptionVec(i).is_inf && !io.in.SignVec(i)))
    NInfVec := Cat((0 until (ReduceWidth/8 + 1)).map(i => io.in.ExceptionVec(i).is_inf && io.in.SignVec(i)))
    NZeroVec := Cat((0 until (ReduceWidth/8 + 1)).map(i => io.in.ExceptionVec(i).is_zero && io.in.SignVec(i)))

    val HasNaN = NaNVec.orR
    val HasPInf = PInfVec.orR
    val HasNInf = NInfVec.orR
    val OnlyNZero = NZeroVec.andR

    io.out.SumException := Cat(HasNaN, HasPInf, HasNInf, OnlyNZero)
    //尾数右移
    // val ProductRShift = Wire(Vec(ReduceWidth/8 + 1, SInt(26.W)))

    // ProductRShift(ReduceWidth/16) := Mux(io.in.CMantissa < 0.S, (-((- io.in.CMantissa(25, 0).asSInt) >> RightShiftVec(ReduceWidth/16))),
    //     (io.in.CMantissa(25, 0).asSInt >> RightShiftVec(ReduceWidth/16)))

    // ProductRShift(ReduceWidth/8) := io.in.CMantissa(25, 0).asSInt >> io.in.RightShiftVec(ReduceWidth/8)

    //这里的问题是，如果c是负数，c为补码表示，右移完之后会变成全F，取负再相加后会变成1.与右移完之后变成0的情况不符
    // printf("ProductRShift_C: %x\n", ProductRShift(ReduceWidth/16))
    
    // for(i <- 0 until ReduceWidth/16){
    //     val tempProduct0 = Wire(SInt(26.W))
    //     tempProduct0 := Cat(io.in.Product0(i)(22, 0), 0.U(3.W)).asSInt
    //     // ProductRShift(i) := Mux(tempProduct0 < 0.S, -((-tempProduct0) >> RightShiftVec(i)), ((tempProduct0) >> RightShiftVec(i)))
    //     ProductRShift(i) := tempProduct0 >> io.in.RightShiftVec(i)
    //     // printf("ProductRShift[%d]: %x\n", i.U,ProductRShift(i))
    // }

    // for(i <- 0 until ReduceWidth/16){
    //     val tempProduct1 = Wire(SInt(26.W))
    //     tempProduct1 := Cat(io.in.Product1(i)(8, 0), 0.U(17.W)).asSInt
    //     ProductRShift(ReduceWidth/16 + i) := tempProduct1 >> io.in.RightShiftVec(ReduceWidth/16 + i)
    // }

    // for(i <- 0 until ReduceWidth/16 + 1){
    //     if(i != ReduceWidth/16){
    //         io.out.Product0(i) := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U, 
    //             io.in.Product0(i).pad(26), ProductRShift(i))
    //     } else {
    //         io.out.Product0(i) := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U, 
    //             0.S(26.W), ProductRShift(ReduceWidth/8))
    //     }
    // }

    // for (i <- 0 until ReduceWidth/16){
    //     io.out.Product1(i) := Mux(io.in.opcode === 7.U, ProductRShift(ReduceWidth/16 + i), io.in.Product1(i))
    // }

    // FP4Reduce sum
    for (i <- 0 until ReduceWidth/4/16){
        io.out.FP4ReduceRes(i) := io.in.FP4ReduceRes.slice(i * FP4P1AddNum, (i + 1) * FP4P1AddNum).reduce(_ + _)
        if (DEBUG_FP4) {
            printf("FP4ReduceRes[%d]: %x\n", i.U, io.out.FP4ReduceRes(i))
        }
    }

    io.out.Product0 := io.in.Product0
    io.out.Product1 := io.in.Product1
    io.out.CMantissa := io.in.CMantissa
    io.out.MaxExp := productMaxExp
    io.out.opcode := io.in.opcode
    io.out.SignVec := io.in.SignVec
    io.out.scaleFP := io.in.scaleFP
}

class FReduceMACPipe2(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe1Result)
        val out = Output(new FPipe2Result)
    })

    val Product1 = Wire(Vec(ReduceWidth/16, SInt(26.W)))
    val Product0 = Wire(Vec(ReduceWidth/16 + 1, SInt(26.W))) //每个向量的尾数右移结果

    val ProductRShift = Wire(Vec(ReduceWidth/8 + 1, SInt(26.W)))

    // ProductRShift(ReduceWidth/16) := Mux(io.in.CMantissa < 0.S, (-((- io.in.CMantissa(25, 0).asSInt) >> RightShiftVec(ReduceWidth/16))),
    //     (io.in.CMantissa(25, 0).asSInt >> RightShiftVec(ReduceWidth/16)))

    ProductRShift(ReduceWidth/8) := io.in.CMantissa(25, 0).asSInt >> io.in.RightShiftVec(ReduceWidth/8)

    //这里的问题是，如果c是负数，c为补码表示，右移完之后会变成全F，取负再相加后会变成1.与右移完之后变成0的情况不符
    if (DEBUG_FP8)
    {printf("ProductRShift_C: %x\n", ProductRShift(ReduceWidth/16))}
    
    for(i <- 0 until ReduceWidth/16){
        val tempProduct0 = Wire(SInt(26.W))
        tempProduct0 := Cat(io.in.Product0(i)(22, 0), 0.U(3.W)).asSInt
        // ProductRShift(i) := Mux(tempProduct0 < 0.S, -((-tempProduct0) >> RightShiftVec(i)), ((tempProduct0) >> RightShiftVec(i)))
        ProductRShift(i) := tempProduct0 >> io.in.RightShiftVec(i)
        if (DEBUG_FP8)
        {printf("ProductRShift[%d]: %x\n", i.U,ProductRShift(i))}
    }

    for(i <- 0 until ReduceWidth/16){
        val tempProduct1 = Wire(SInt(26.W))
        tempProduct1 := Cat(io.in.Product1(i)(22, 0), 0.U(3.W)).asSInt
        ProductRShift(ReduceWidth/16 + i) := tempProduct1 >> io.in.RightShiftVec(ReduceWidth/16 + i)
        if (DEBUG_FP8)
        {printf("ProductRShift[%d]: %x\n", (i + ReduceWidth/16).U,ProductRShift(ReduceWidth/16 + i))}
    }

    for(i <- 0 until ReduceWidth/16 + 1){
        if(i != ReduceWidth/16){
            Product0(i) := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U, 
                io.in.Product0(i).pad(26), ProductRShift(i))
        } else {
            Product0(i) := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U, 
                0.S(26.W), ProductRShift(ReduceWidth/8))
        }
    }

    for (i <- 0 until ReduceWidth/16){
        Product1(i) := Mux(io.in.opcode === 7.U || io.in.opcode === 8.U || io.in.opcode === 11.U || io.in.opcode === 12.U, ProductRShift(ReduceWidth/16 + i), io.in.Product1(i))
    }

    // Phase1 of fp4 max
    val FP4MaxExp = Wire(UInt(9.W))
    val cmptreefp4p1 = Module(new CmpTreeP1(8, ReduceWidth/4/MinGroupSize + 1, 9))
    cmptreefp4p1.io.in := io.in.CmpTreefp4P0Result
    FP4MaxExp := cmptreefp4p1.io.out


    // Shift for FP4
    val FP4Shift = Wire(Vec(ReduceWidth/4/MinGroupSize + 1, UInt(9.W)))
    for (i <- 0 until ReduceWidth/4/MinGroupSize + 1){
        FP4Shift(i) := FP4MaxExp.asUInt - io.in.scaleFP(i).exp.asUInt
    }
    // FP4Shift(ReduceWidth/4/MinGroupSize) := io.in.FP4MaxExp.asUInt - io.in.CExp(7, 0).asUInt

    for (i <- 0 until ReduceWidth/4/MinGroupSize){
        val tempFP4 = Wire(SInt((32 + 3).W)) // 32位，   向左保留3位，减少负数右移遗留-1累积的影响
        tempFP4 := (io.in.FP4ReduceRes(i) * io.in.scaleFP(i).signed_sig).pad(32 + 3) << Mux(io.in.opcode === 10.U, (15 + 3).U, (6 + 3).U)
        io.out.FP4ABShift(i) := tempFP4 >> FP4Shift(i)
        if (DEBUG_FP4) {
            printf("FP4Shift[%d]: %x\n", i.U, FP4Shift(i))
            printf("tempFP4[%d]: %x\n", i.U, tempFP4)
            printf("io.out.FP4ABShift[%d]: %x\n", i.U, io.out.FP4ABShift(i))
            printf("without pad io.out.FP4ABShift[%d]: %x\n", i.U, io.out.FP4ABShift(i)(34, 3))
        }
    }
    if (DEBUG_FP4) {
        printf("FP4Shift[8]: %x\n", FP4Shift(ReduceWidth/4/MinGroupSize))
        printf("io.out.FP4ABShift[8]: %x\n", io.out.FP4ABShift(ReduceWidth/4/MinGroupSize))
    }
    val FP4signedCMantissa = Wire(SInt((32 + 3).W))
    FP4signedCMantissa := Cat(io.in.CMantissa, 0.U(3.W)).asSInt >> FP4Shift(ReduceWidth/4/MinGroupSize)
    io.out.FP4ABShift(ReduceWidth/4/MinGroupSize) := Mux(io.in.SignVec(ReduceWidth/8), -FP4signedCMantissa, FP4signedCMantissa)

    val ProductRShiftF = Wire(Vec(ReduceWidth/16, SInt((26 + log2Ceil(P2AddNum)).W)))
    val Int8Product = Wire(Vec(ReduceWidth/16, SInt((26 + log2Ceil(P2AddNum)).W)))

    for(i <- 0 until ReduceWidth/16){
        // ProductRShift(i) := io.in.Product0(i).pad(26 + log2Ceil(P2AddNum))
        ProductRShiftF(i) := Mux(io.in.SignVec(i), -Product0(i), Product0(i)).pad(26 + log2Ceil(P2AddNum))
    }

    for(i <- 0 until ReduceWidth/16){
        Int8Product(i) := Mux(io.in.SignVec(ReduceWidth/16 + i), -Product1(i), Product1(i)).pad(26 + log2Ceil(P2AddNum))
    }
    // printf("Int8Product0:%x\n", Int8Product(0))


    val SumResult0 = Wire(Vec(P3AddNum, SInt((26 + log2Ceil(P2AddNum)).W)))
    val SumResult1 = Wire(Vec(P3AddNum, SInt((26 + log2Ceil(P2AddNum)).W)))
    // val SumResult0 = Wire(Vec(4, SInt(32.W)))
    // val SumResult1 = Wire(Vec(4, SInt(32.W)))
    for(i <- 0 until P3AddNum) {
        SumResult0(i) := ProductRShiftF.slice(i * P2AddNum, (i + 1) * P2AddNum).reduce(_ + _)
        SumResult1(i) := Int8Product.slice(i * P2AddNum, (i + 1) * P2AddNum).reduce(_ + _)
    }

    io.out.ReduceRes0 := SumResult0
    io.out.ReduceRes1 := SumResult1
    io.out.CMantissa := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U, 
        io.in.CMantissa, Mux(io.in.SignVec(ReduceWidth/8), -Product0(ReduceWidth/16), Product0(ReduceWidth/16)).pad(32))
    io.out.MaxExp := Mux(io.in.opcode === 9.U || io.in.opcode === 10.U, FP4MaxExp.asUInt.pad(10) + 256.U, io.in.MaxExp)
    if (DEBUG_FP4) {
        printf("FP4MaxExp: %x\n", FP4MaxExp)
        printf("io.out.MaxExp: %x\n", io.out.MaxExp)
    }
    io.out.opcode := io.in.opcode
    io.out.SumException := io.in.SumException
}

class FReduceMACPipe3(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe2Result)
        val out = Output(new FPipe3Result)
    })

    // val ReduceResFP4 = Wire(SInt(32.W))
    // ReduceResFP4 := io.in.FP4ABShift.reduce(_ + _)

    val SumResult0 = Wire(Vec(P3AddNum, SInt((32 + 3).W)))
    val SumResult1 = Wire(Vec(P3AddNum, SInt((32).W)))
    for(i <- 0 until P3AddNum) {
        SumResult0(i) := Mux(io.in.opcode === 9.U || io.in.opcode === 10.U, io.in.FP4ABShift(i), Cat(io.in.ReduceRes0(i).pad(32), 0.U(3.W)).asSInt)
        SumResult1(i) := io.in.ReduceRes1(i).pad(32)
    }

    
    io.out.ReduceRes0 := (SumResult0.reduce(_ + _) + Mux(io.in.opcode === 9.U || io.in.opcode === 10.U, io.in.FP4ABShift(ReduceWidth / 4 / MinGroupSize), Cat(io.in.CMantissa, 0.U(3.W)).asSInt))(34, 3).asSInt
    
    io.out.ReduceRes1 := SumResult1.reduce(_ + _)

    if (DEBUG_FP8)
    {printf("CMantissa: %x\n", io.in.CMantissa)}


    // io.out.ReduceRes := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U || io.in.opcode === 7.U, 
    //     ReduceResI, ReduceResF)
    // printf("ReduceResI: %x\n", ReduceResI)
    io.out.CMantissa := io.in.CMantissa
    io.out.MaxExp := io.in.MaxExp
    io.out.opcode := io.in.opcode
    io.out.SumException := io.in.SumException
    // io.out.ReduceResFP4 := ReduceResFP4
}

class FReduceMACPipe4(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe3Result)
        val out = UInt(ResultWidth.W)
    })

    val ReduceResF = Wire(SInt(32.W))
    val ReduceResI = Wire(SInt(32.W))
    val ReduceRes  = Wire(SInt(32.W))
    val SumResult0 = Wire(SInt((32).W))
    val SumResult1 = Wire(SInt((32).W))
    
    SumResult0 := io.in.ReduceRes0.pad(32) //.pad(26 + log2Ceil(ReduceWidth/16))
    SumResult1 := io.in.ReduceRes1.pad(32)


    ReduceRes := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U || io.in.opcode === 7.U || io.in.opcode === 8.U || io.in.opcode === 11.U || io.in.opcode === 12.U, 
        SumResult0 + SumResult1.pad(32), SumResult0)

    if (DEBUG_FP8 || DEBUG_FP4)
    {
        printf("SumResult0:%x\n", SumResult0)
        printf("SumResult1:%x\n", SumResult1)
        printf("ReduceRes:%x\n", ReduceRes)
    }

    ReduceResF := ReduceRes
    ReduceResI := ReduceRes

    val ExceptionBits = Wire(UInt(32.W))
    val IsException = Wire(Bool())
    IsException := io.in.SumException.orR || (ReduceRes === 0.S)

    ExceptionBits := Mux(io.in.SumException(3) || io.in.SumException(2) && io.in.SumException(1), 0x7FC00000.U,
        Mux(io.in.SumException(2), 0x7F800000.U, 
            Mux(io.in.SumException(1), BigInt("FF800000", 16).U, 0.U)
        )
    )

    //尾数符号位
    val ResultSign = Wire(Bool())
    ResultSign := ReduceResF(31)

    //去掉符号位，正直接转无符号数，负先有符号数取负再转无符号数
    val UnsignedSig = Wire(UInt(32.W))
    UnsignedSig := Mux(ResultSign, 
        (-ReduceResF.asSInt).asUInt, 
        ReduceResF.asUInt
    )
    if (DEBUG_FP8)
    {printf("UnsignedSig:%x\n", UnsignedSig)}

    val clz = Module(new CLZ(32))
    clz.io.in := UnsignedSig
    val LeadingZeros = Wire(UInt(5.W))
    LeadingZeros := clz.io.out
    if (DEBUG_FP8)
    {printf("LZ:%d\n", LeadingZeros)}
    //左移相同，该怎么左移就怎么左移,注意要把最高位的1省略掉作为规格化结果

    //修改一下这里规格化的逻辑
    //真正问题在乘法，乘法尾数是0，但阶码正常计算了
    //最终导致阶码比大小的时候，阶码较大的数尾数是0，不参与加法，而阶码第二大的数，尾数又要右移，导致丢精度，例如fp16_xtest.txt中的数据
    val ShiftMantissa = Wire(SInt(32.W))
    val ShiftExp = Wire(SInt(16.W))

    ShiftExp := (io.in.MaxExp - 511.U(10.W)).asSInt.pad(16) - LeadingZeros.pad(16).asSInt + 8.S(16.W)
    ShiftMantissa := Mux(LeadingZeros > 8.U, UnsignedSig.asSInt << (LeadingZeros - 8.U),
        UnsignedSig.asSInt >> (8.U - LeadingZeros) //这里右移导致移掉了低位的非零位，丢失信息，需要改
    )

    if (DEBUG_FP4) {
        printf("ShiftExp:%x\n", ShiftExp)
    }

    if (DEBUG_FP8 || DEBUG_FP4)
    {printf("ShiftMantissa:%x\n", ShiftMantissa)}

    val ResultExp = Wire(UInt(8.W))
    val ResultSig = Wire(UInt(24.W))
    ResultExp := Mux(ShiftExp < -126.S(16.W), 0.U, 
        Mux(ShiftExp > 127.S(16.W), 255.U, (ShiftExp + 127.S(16.W)).asUInt)
    )
    ResultSig := Mux(ShiftExp < -149.S(16.W), 0.U, 
        Mux(ShiftExp < -126.S(16.W), ShiftMantissa(23, 0).asUInt >> (-126.S(16.W) - ShiftExp).asUInt,
            Mux(ShiftExp > 127.S(16.W), 0.U, ShiftMantissa(23, 0).asUInt)
        )
    )

    val Result = Wire(UInt(ResultWidth.W))
    if (DEBUG_FP8 || DEBUG_FP4)
    {
        printf("Result:%x\n", Result)
    }
    Result := Cat(ResultSign, ResultExp, ResultSig(22, 0))
    io.out := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U, 
        ReduceResI.asUInt, Mux(IsException, ExceptionBits, Result)
    )
}

// class top extends CuteModule {
//     val io = IO(new Bundle{
//         val AVector = Flipped(DecoupledIO(UInt(ReduceWidth.W)))
//         val BVector = Flipped(DecoupledIO(UInt(ReduceWidth.W)))
//         val CAdd    = Flipped(DecoupledIO(UInt(ResultWidth.W)))
//         val DResult = DecoupledIO(UInt(ResultWidth.W))
//         val opcode = Input(UInt(3.W))   //0:Int8, 1:FP16, 2:BF16, 3:TF32
//     })

//     val pipe0 = Module(new FReduceMACPipe0)
//     val pipe1 = Module(new FReduceMACPipe1)
//     val pipe2 = Module(new FReduceMACPipe2)
//     val pipe3 = Module(new FReduceMACPipe3)


//     pipe0.io.inA := io.AVector.bits
//     pipe0.io.inB := io.BVector.bits
//     pipe0.io.inC := io.CAdd.bits
//     pipe0.io.opcode := io.opcode

//     io.AVector.ready := true.B
//     io.BVector.ready := true.B
//     io.CAdd.ready := true.B

//     pipe1.io.in := pipe0.io.out
//     pipe2.io.in := pipe1.io.out
//     pipe3.io.in := pipe2.io.out

//     io.DResult.bits := pipe3.io.out
//     io.DResult.valid := true.B

// }

//这里的top是切好流水的，为方便debug，先写一个单周期的top

class FReducePE(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val AVector = Flipped(DecoupledIO(UInt(ReduceWidth.W)))
        val BVector = Flipped(DecoupledIO(UInt(ReduceWidth.W)))
        val CAdd    = Flipped(DecoupledIO(UInt(ResultWidth.W)))
        val AScale  = Flipped(DecoupledIO(UInt((ReduceWidth/MinDataTypeWidth/MinGroupSize * ScaleElementWidth).W)))
        val BScale  = Flipped(DecoupledIO(UInt((ReduceWidth/MinDataTypeWidth/MinGroupSize * ScaleElementWidth).W)))
        val DResult = DecoupledIO(UInt(ResultWidth.W))
        val opcode = Input(UInt(4.W))   //0:Int8, 1:FP16, 2:BF16, 3:TF32, 4:I8 * UI8, 5:UI8 * I8, 6:UI8 * UI8, 7:MXFP8E4M3, 8:MXFP8e5m2 9:NVFP4, 10:MXFP4, 11:FP8E4M3, 12:FP8E5M2
    })

    // for DEBUG
    val count = RegInit(0.U(32.W))
    count := count + 1.U

    val pipe0 = Module(new FReduceMACPipe0)
    val pipe1 = Module(new FReduceMACPipe1)
    val pipe2 = Module(new FReduceMACPipe2)
    val pipe3 = Module(new FReduceMACPipe3)
    val pipe4 = Module(new FReduceMACPipe4)

    if (DEBUG_FP8 || DEBUG_FP4) {
        printf("Cycle count: %x\n", count)

        printf("io.AVector.bits: %x\n", io.AVector.bits)
        printf("io.BVector.bits: %x\n", io.BVector.bits)
        printf("op.AScale.bits[0]: %x\n", io.AScale.bits(7, 0))
        printf("op.BScale.bits[0]: %x\n", io.BScale.bits(7, 0))

        printf("AScale: %x\n", io.AScale.bits)
    }


    val PipeResRegValid = RegInit(VecInit(Seq.fill(6)(false.B)))

    val InputRegC = Reg(UInt((ResultWidth + 4).W))
    val InputRegA = Reg(new FDecodeResult)
    val InputRegB = Reg(new FDecodeResult)
    val InputRegAFP4Vec = Reg(Vec(ReduceWidth/4, SInt(5.W)))
    val InputRegBFP4Vec = Reg(Vec(ReduceWidth/4, SInt(5.W)))
    val InputRegAscale = Reg(Vec(ReduceWidth/MinDataTypeWidth/MinGroupSize, UInt(ScaleElementWidth.W)))
    val InputRegBscale = Reg(Vec(ReduceWidth/MinDataTypeWidth/MinGroupSize, UInt(ScaleElementWidth.W)))
    val Pipe0ResReg = Reg(new FPipe0Result)
    val Pipe1ResReg = Reg(new FPipe1Result)
    val Pipe2ResReg = Reg(new FPipe2Result)
    val Pipe3ResReg = Reg(new FPipe3Result)
    val Pipe4ResReg = Reg(UInt(ResultWidth.W))

    val InReady = Wire(Bool())
    val Pipe0ResRegAllowIn = Wire(Bool())
    val Pipe1ResRegAllowIn = Wire(Bool())
    val Pipe2ResRegAllowIn = Wire(Bool())
    val Pipe3ResRegAllowIn = Wire(Bool())
    val Pipe4ResRegAllowIn = Wire(Bool())

    Pipe4ResRegAllowIn := (!PipeResRegValid(5)) || io.DResult.ready
    Pipe3ResRegAllowIn := (!PipeResRegValid(4)) || Pipe4ResRegAllowIn
    Pipe2ResRegAllowIn := (!PipeResRegValid(3)) || Pipe3ResRegAllowIn
    Pipe1ResRegAllowIn := (!PipeResRegValid(2)) || Pipe2ResRegAllowIn
    Pipe0ResRegAllowIn := (!PipeResRegValid(1)) || Pipe1ResRegAllowIn
    InReady := (!PipeResRegValid(0)) || Pipe0ResRegAllowIn

    io.AVector.ready := InReady
    io.BVector.ready := InReady
    io.AScale.ready := InReady
    io.BScale.ready := InReady
    io.CAdd.ready := InReady

    val ABCValid = Wire(UInt(1.W))
    ABCValid := io.AVector.valid && io.BVector.valid && io.CAdd.valid && ((io.opcode =/= 7.U && io.opcode =/= 8.U && io.opcode =/= 9.U && io.opcode =/= 10.U) || (io.AScale.valid && io.BScale.valid))

    val ADecoder = Module(new FVecDecoder)
    val BDecoder = Module(new FVecDecoder)

    ADecoder.io.in := io.AVector.bits
    ADecoder.io.opcode := Mux(io.opcode === 4.U || io.opcode === 0.U, 0.U, 
        Mux(io.opcode === 5.U || io.opcode === 6.U, 4.U, io.opcode))
    BDecoder.io.in := io.BVector.bits
    BDecoder.io.opcode := Mux(io.opcode === 5.U || io.opcode === 0.U, 0.U, 
        Mux(io.opcode === 4.U || io.opcode === 6.U, 4.U, io.opcode))

    // 在MXFP8时提前将scale相加加到A向量的exp上
    val scaleSum = Wire(Vec(ReduceWidth/8/32, UInt(9.W)))
    for (i <- 0 until ReduceWidth/8/32) {
        scaleSum(i) := io.AScale.bits(i * 8 + 7, i * 8).asUInt.pad(10) + io.BScale.bits(i * 8 + 7, i * 8).asUInt.pad(10) - 254.U
    }


    when(InReady){
        when(ABCValid === 1.U){
            InputRegC := Cat(io.CAdd.bits, io.opcode)
            InputRegA := ADecoder.io.out
            InputRegB := BDecoder.io.out
            for (i <- 0 until ReduceWidth/8/32){
                // printf("scaleSum[%d]: %x\n", i.U, scaleSum(i).asSInt.pad(10))
                for (j <- 0 until 32){
                    InputRegA.TF32Vec(i * 32 + j).exp := Mux(io.opcode === 7.U || io.opcode === 8.U, scaleSum(i).asSInt.pad(10) + ADecoder.io.out.TF32Vec(i * 32 + j).exp.pad(10), ADecoder.io.out.TF32Vec(i * 32 + j).exp.pad(10))
                }
            }
            InputRegAscale := io.AScale.bits.asTypeOf(InputRegAscale)
            InputRegBscale := io.BScale.bits.asTypeOf(InputRegBscale)
            PipeResRegValid(0) := true.B
        }.otherwise{
            InputRegC := InputRegC
            PipeResRegValid(0) := false.B
        }
    }.otherwise{
        InputRegC := InputRegC
        PipeResRegValid(0) := PipeResRegValid(0)
    }

    // FP4 to int
    for (i <- 0 until ReduceWidth/4){
        val fp4DecodeA = Module(new FP4toint)
        val fp4DecodeB = Module(new FP4toint)
        fp4DecodeA.io.in := io.AVector.bits((i+1)*4-1, i*4)
        fp4DecodeB.io.in := io.BVector.bits((i+1)*4-1, i*4)
        when(InReady && ABCValid === 1.U && (io.opcode === 9.U || io.opcode === 10.U)){
            InputRegAFP4Vec(i) := fp4DecodeA.io.out
            InputRegBFP4Vec(i) := fp4DecodeB.io.out
        }
    }
    
    pipe0.io.inA := InputRegA
    pipe0.io.inB := InputRegB
    pipe0.io.inAFP4Vec := InputRegAFP4Vec
    pipe0.io.inBFP4Vec := InputRegBFP4Vec
    pipe0.io.inC := InputRegC(ResultWidth + 3, 4)
    pipe0.io.inAscale := InputRegAscale
    pipe0.io.inBscale := InputRegBscale
    pipe0.io.opcode := InputRegC(3, 0)

    when(Pipe0ResRegAllowIn){
        when(PipeResRegValid(0)){
            PipeResRegValid(1) := true.B
            Pipe0ResReg := pipe0.io.out
        }.otherwise{
            PipeResRegValid(1) := false.B
            Pipe0ResReg := Pipe0ResReg
        }
    }.otherwise{
        PipeResRegValid(1) := PipeResRegValid(1)
        Pipe0ResReg := Pipe0ResReg
    }

    pipe1.io.in := Pipe0ResReg

    when(Pipe1ResRegAllowIn){
        when(PipeResRegValid(1)){
            PipeResRegValid(2) := true.B
            Pipe1ResReg := pipe1.io.out
        }.otherwise{
            PipeResRegValid(2) := false.B
            Pipe1ResReg := Pipe1ResReg
        }
    }.otherwise{
        PipeResRegValid(2) := PipeResRegValid(2)
        Pipe1ResReg := Pipe1ResReg
    }

    pipe2.io.in := Pipe1ResReg

    when(Pipe2ResRegAllowIn){
        when(PipeResRegValid(2)){
            PipeResRegValid(3) := true.B
            Pipe2ResReg := pipe2.io.out
        }.otherwise{
            PipeResRegValid(3) := false.B
            Pipe2ResReg := Pipe2ResReg
        }
    }.otherwise{
        PipeResRegValid(3) := PipeResRegValid(3)
        Pipe2ResReg := Pipe2ResReg
    }

    pipe3.io.in := Pipe2ResReg

    when(Pipe3ResRegAllowIn){
        when(PipeResRegValid(3)){
            PipeResRegValid(4) := true.B
            Pipe3ResReg := pipe3.io.out
        }.otherwise{
            PipeResRegValid(4) := false.B
            Pipe3ResReg := Pipe3ResReg
        }
    }.otherwise{
        PipeResRegValid(4) := PipeResRegValid(4)
        Pipe3ResReg := Pipe3ResReg
    }

    pipe4.io.in := Pipe3ResReg

    when(Pipe4ResRegAllowIn){
        when(PipeResRegValid(4)){
            PipeResRegValid(5) := true.B
            Pipe4ResReg := pipe4.io.out
        }.otherwise{
            PipeResRegValid(5) := false.B
            Pipe4ResReg := Pipe4ResReg
        }
    }.otherwise{
        PipeResRegValid(5) := PipeResRegValid(5)
        Pipe4ResReg := Pipe4ResReg
    }

    io.DResult.valid := PipeResRegValid(5)
    io.DResult.bits := Pipe4ResReg
}

class top (implicit p: Parameters) extends FReducePE {}

// class FReducePE(id:Int)(implicit p: Parameters) extends CuteModule with HWParameters{
//     val io = IO(new Bundle{
//         val ReduceA = Flipped(DecoupledIO(UInt(ReduceWidth.W)))
//         val ReduceB = Flipped(DecoupledIO(UInt(ReduceWidth.W)))
//         val AddC    = Flipped(DecoupledIO(UInt(ReduceWidth.W)))
//         val ResultD = DecoupledIO(UInt(ResultWidth.W))
//         val ConfigInfo = Flipped((new MTEMicroTaskConfigIO))
//         // val ExternalReduceSize = Flipped(DecoupledIO(UInt(ScaratchpadMaxTensorDimBitSize.W)))
//     })

//     //TODO:init
//     io.ReduceA.ready := false.B
//     io.ReduceB.ready := false.B
//     io.AddC.ready := false.B
//     io.ResultD.valid := false.B
//     io.ResultD.bits := 0.U
//     io.ConfigInfo.ready := false.B


//     //ReducePE和MatrixTE需要一个对于externalReduce的处理，以提高热效率，提供主频，减少对CScratchPad的访问
//     //ExternalReduce是指，我们的Scarchpad内的Tensor的K维大于1时，可以减少从CScratchPad的访问数据，让ReducePE使用自己暂存的累加结果后，再存至CScratchPad
//     //Trick：再来，这里的K越大，我们的CSratchPad的平均访问次数就越少，就可以使用更慢更大的SRAM

//     val FloatReduceMAC = Module(new FReduceMAC)
//     FloatReduceMAC.io.AVector <> io.ReduceA
//     FloatReduceMAC.io.BVector <> io.ReduceB
//     FloatReduceMAC.io.CAdd    <> io.AddC
//     FloatReduceMAC.io.FIFOReady   := false.B
//     //!!!Todo:opcode
//     FloatReduceMAC.io.opcode := io.ConfigInfo.dataType //0:Int8, 1:FP16, 2:BF16, 3:TF32
//     FloatReduceMAC.io.ExternalReduceSize := Tensor_K.U
    

//     //只有在数据类型匹配时才能进行计算
//     //在Reduce内完成数据的握手，及所有数据准备好后才能进行计算，并用一个fifo保存ResultD，等待ResultD被握手
//     val ResultFIFO = RegInit(VecInit(Seq.fill(ResultFIFODepth)(0.U(ResultWidth.W))))
//     val ResultFIFOHead = RegInit(0.U(log2Ceil(ResultFIFODepth).W))
//     val ResultFIFOTail = RegInit(0.U(log2Ceil(ResultFIFODepth).W))
//     val ResultFIFOFull = ResultFIFOTail === WrapInc(ResultFIFOHead, ResultFIFODepth)
//     val ResultFIFOEmpty = ResultFIFOHead === ResultFIFOTail
//     val ResultFIFOValid = WireInit(false.B)


//     //数据类型，整个计算过程中只有一个数据类型，ConfigInfo不会改变
//     val dataType = RegInit(ElementDataType.DataTypeUndef)
//     //PE不工作且FIFO为空时，才能接受新的配置信息
//     val PEWorking = FloatReduceMAC.io.working
//     io.ConfigInfo.ready := !PEWorking && ResultFIFOEmpty
//     when(io.ConfigInfo.valid && io.ConfigInfo.ready){
//         dataType := io.ConfigInfo.dataType
//     //   dataType := ElementDataType.DataTypeSInt8 //默认改成SInt8
//         FloatReduceMAC.io.opcode := Mux(io.ConfigInfo.dataType === ElementDataType.DataTypeSInt8, 0.U,
//             Mux(io.ConfigInfo.dataType === ElementDataType.DataTypeFP16, 1.U,
//                 Mux(io.ConfigInfo.dataType === ElementDataType.DataTypeBF16, 2.U, 3.U)))
//         if(id == 0)
//         {
//             if (YJPDebugEnable)
//             {
//                 printf("[YJPDebug]PE ConfigInfo: %x\n",io.ConfigInfo.dataType)
//                 printf("[YJPDebug]PE Start\n")
//             }
//         }
//     }
    


//     //根据数据类型选择不同的ReduceMAC,作为CurrentResultD的数据源，由于configinfo不会改变，所以这里的DResult不用改变，并设置Valid信号
//     val CurrentResultD = Wire(Valid(UInt(ResultWidth.W)))
//     CurrentResultD := FloatReduceMAC.io.DResult

//     when(CurrentResultD.valid){
//         when(!ResultFIFOFull){
//             ResultFIFO(ResultFIFOHead) := CurrentResultD.bits
//             when(ResultFIFOHead+1.U===ResultFIFODepth.U){
//                 ResultFIFOHead := 0.U
//             }.otherwise{
//                 ResultFIFOHead := ResultFIFOHead + 1.U
//             }
//             if(id == 0)
//             {
//                 if (YJPDebugEnable)
//                 {
//                     printf("[YJPDebug]PE ResultFIFO Insert: %x\n",CurrentResultD.bits)
//                 }
//             }
//         }.otherwise{
//         // printf(p"ResultFIFOFull\n")
//             if(id == 0)
//             {
//                 if (YJPDebugEnable)
//                 {
//                     printf("[YJPDebug]PE FIFO ResultFIFOFull\n")
//                 }
//             }
//         }
//     }


//     when(ResultFIFOEmpty){
//         ResultFIFOValid := false.B
//     }.otherwise{
//         io.ResultD.bits := ResultFIFO(ResultFIFOTail)
//         ResultFIFOValid := true.B
//         io.ResultD.valid := true.B
//         when(io.ResultD.fire){
//             when(ResultFIFOTail+1.U===ResultFIFODepth.U){
//                 ResultFIFOTail := 0.U
//             }.otherwise{
//                 ResultFIFOTail := ResultFIFOTail + 1.U
//             }
//             if(id == 0)
//             {
//                 if (YJPDebugEnable)
//                 {
//                     printf("[YJPDebug]PE ResultFIFO Pop: %x\n",io.ResultD.bits)
//                 }
//             }
//         }
//     }


//     //数据源ReduceA ReduceB AddC什么时候能置ready？
//     //全部valid的时候才可以，同时当前流水下的所有数据都能在fifo中存的下，才能置ready
//     //方案1:已知MACTree的流水线深度，已知ResultFIFO的深度，可以得出ResultFIFO存的数据达到某个深度时，可以安全的接受新的数据
//     //方案2：直接用FIFO满没满确定是否ready，整体流水线都受这个制约，好像有点粗暴？只要ready，所有数据往下流一个流水级，否则不动
//     val InputReady = ResultFIFOFull===false.B
//     io.ReduceA.ready := InputReady
//     io.ReduceB.ready := InputReady
//     io.AddC.ready    := InputReady

//     //什么时候能接让MacTree的数据输入到fifo？
//     //MacTree的数据输入到fifo的时候，fifo不满，且MacTree的数据有效
//     val MacTreeReady = ResultFIFOFull===false.B
//     FloatReduceMAC.io.FIFOReady := MacTreeReady
//     //输出的ResultD什么时候能置valid？
//     //ResultFIFO不为空时，才能置valid
//     // io.ResultD.valid := ResultFIFOValid

// }

