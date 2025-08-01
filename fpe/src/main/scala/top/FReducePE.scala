//记得修改if相关，改成mux
package cute

import chisel3._
import chisel3.util._

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
    def fromUInt(x: UInt, expWidth: Int, pc: Int): RawFloat = {
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
      fp.exception.is_nan := x(expWidth + pc - 2, pc - 1).andR && x(pc - 2, 0).orR
      fp.exception.is_inf := x(expWidth + pc - 2, pc - 1).andR && !x(pc - 2, 0).orR
      fp.exception.is_zero := !x(expWidth + pc - 2, 0).orR
      fp
    }
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
    val Int8Vec = Vec(ReduceWidth/8, UInt(9.W))
    val TF32Vec = Vec(ReduceWidth/16, new RawFloat(8, 11))
}

class FVecDecoder(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(UInt(ReduceWidth.W))
        val opcode = Input(UInt(3.W))   //0:Int8, 1:FP16, 2:BF16, 3:TF32, 4:UI8
        val out = Output(new FDecodeResult)
    })

    for(i <- 0 until ReduceWidth/8){
        io.out.Int8Vec(i) := Mux(io.opcode === 0.U, 
            io.in(8 * i + 7, 8 * i).asSInt.pad(9).asUInt, io.in(8 * i + 7, 8 * i).pad(9).asUInt
        )
    }

    for(i <- 0 until ReduceWidth/32){
        val Bits16 = io.in(16 * i + 15, 16 * i)
        val Bits32 = io.in(32 * i + 31, 32 * i)
        val DecodeFP16 = RawFloat.fromUInt(Bits16, 5, 11)
        val DecodeBF16 = RawFloat.fromUInt(Bits16, 8, 8)
        val FP16toTF32 = Wire(new RawFloat(8, 11))
        val BF16toTF32 = Wire(new RawFloat(8, 11))
        FP16toTF32.exp := DecodeFP16.exp.pad(8)
        FP16toTF32.signed_sig := DecodeFP16.signed_sig
        FP16toTF32.exception := DecodeFP16.exception
        FP16toTF32.sign := DecodeFP16.sign
        BF16toTF32.exp := DecodeBF16.exp
        BF16toTF32.signed_sig := Cat(DecodeBF16.signed_sig, 0.U(3.W)).asSInt
        BF16toTF32.exception := DecodeBF16.exception
        BF16toTF32.sign := DecodeBF16.sign
        io.out.TF32Vec(i) := Mux(
            io.opcode === 1.U, FP16toTF32,
            Mux(io.opcode === 2.U, BF16toTF32,
                RawFloat.fromUInt(Bits32(31, 13), 8, 11)
            )
        )
    }

    val TF32Zero = RawFloat.fromUInt(0.U(19.W), 8, 11)

    for(i <- ReduceWidth/32 until ReduceWidth/16){
        val Bits16 = io.in(16 * i + 15, 16 * i)
        val DecodeFP16 = RawFloat.fromUInt(Bits16, 5, 11)
        val DecodeBF16 = RawFloat.fromUInt(Bits16, 8, 8)
        val FP16toTF32 = Wire(new RawFloat(8, 11))
        val BF16toTF32 = Wire(new RawFloat(8, 11))
        FP16toTF32.exp := DecodeFP16.exp.pad(8)
        FP16toTF32.signed_sig := DecodeFP16.signed_sig
        FP16toTF32.exception := DecodeFP16.exception
        FP16toTF32.sign := DecodeFP16.sign
        BF16toTF32.exp := DecodeBF16.exp
        BF16toTF32.signed_sig := Cat(DecodeBF16.signed_sig, 0.U(3.W)).asSInt// 有符号数，所以12位
        BF16toTF32.exception := DecodeBF16.exception
        BF16toTF32.sign := DecodeBF16.sign
        io.out.TF32Vec(i) := Mux(
            io.opcode === 1.U, FP16toTF32,
            Mux(io.opcode === 2.U, BF16toTF32,
                TF32Zero
            )
        )
    }


}

class CmpTree(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle {
        val in    = Input(Vec(ReduceWidth/16 + 1, UInt(9.W)))
        val out   = Output(UInt(9.W))
    })

  // 生成 in_vec_0: 9组，每组ReduceWidth/16位
    val inVec0 = Wire(Vec(9, UInt((ReduceWidth/16 + 1).W)))
    for (j <- 0 until 9) {
        inVec0(j) := Cat((0 until ReduceWidth/16 + 1).reverse.map(i => io.in(i)(8 - j)))
    } 

  // 生成 mask: 9组，每组ReduceWidth/16位
    val mask = Wire(Vec(9, UInt((ReduceWidth/16 + 1).W)))
    mask(0) := Mux(inVec0(0) === 0.U, ~inVec0(0), inVec0(0))
    for (i <- 1 until 9) {
        val prev = Wire(UInt((ReduceWidth/16 + 1).W))
        val curr = Wire(UInt((ReduceWidth/16 + 1).W))
        val prevandcurr = Wire(UInt((ReduceWidth/16 + 1).W))
        prev := mask(i - 1)
        curr := inVec0(i)
        prevandcurr := (prev & curr)
        mask(i) := Mux(prevandcurr === 0.U, prev, prevandcurr)
    }

  // 生成 out: 9位
    io.out := Cat((0 until 9).map(k => (inVec0(k) & mask(8)).orR))
}

class CmpTreeP0Res(layers:Int)(implicit p: Parameters) extends CuteBundle{
    val part1   = UInt((ReduceWidth/16 + 1).W)   //第6层mask（也就是mask(5)）,用于算后续的mask
    val part2   = Vec(9 - layers, UInt((ReduceWidth/16 + 1).W)) //6层以后的数据传到下一层
    val part3   = UInt(layers.W) //前6层的结果
}

class CmpTreeP0(layers:Int)(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle {
        val in    = Input(Vec(ReduceWidth/16 + 1, UInt(9.W)))
        val out   = Output(new CmpTreeP0Res(layers))
    })

  // 生成 in_vec_0: 9组，每组ReduceWidth/16位
    val inVec0 = Wire(Vec(9, UInt((ReduceWidth/16 + 1).W)))
    for (j <- 0 until 9) {
        inVec0(j) := Cat((0 until ReduceWidth/16 + 1).reverse.map(i => io.in(i)(8 - j)))
    } 

  // 生成 mask: 9组，每组ReduceWidth/16位
    val mask = Wire(Vec(layers, UInt((ReduceWidth/16 + 1).W)))
    mask(0) := Mux(inVec0(0) === 0.U, ~inVec0(0), inVec0(0))
    for (i <- 1 until layers) {
        val prev = Wire(UInt((ReduceWidth/16 + 1).W))
        val curr = Wire(UInt((ReduceWidth/16 + 1).W))
        val prevandcurr = Wire(UInt((ReduceWidth/16 + 1).W))
        prev := mask(i - 1)
        curr := inVec0(i)
        prevandcurr := (prev & curr)
        mask(i) := Mux(prevandcurr === 0.U, prev, prevandcurr)
    }

  // 生成 out: 9位
    io.out.part1 := mask(layers - 1)
    for(i <- 0 until (9 - layers)){
        io.out.part2(i) := inVec0(i + layers)
    }

    val res = Wire(Vec(layers, UInt(1.W)))
    for(i <- 0 until layers){
        val bit = Wire(UInt((ReduceWidth/16 + 1).W))
        bit := mask(layers - 1) & inVec0(i)
        res(i) := Mux(bit === 0.U, 0.U, 1.U)
    }

    io.out.part3 := Cat((0 until layers).map(k => res(k)))
}

class CmpTreeP1(layers : Int)(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle {
        val in    = Input(new CmpTreeP0Res(layers))
        val out   = Output(UInt(9.W))
    })

    val mask = Wire(Vec(9 - layers, UInt((ReduceWidth/16 + 1).W)))
    for (i <- 0 until (9 - layers)) {
        val prev = Wire(UInt((ReduceWidth/16 + 1).W))
        val curr = Wire(UInt((ReduceWidth/16 + 1).W))
        val prevandcurr = Wire(UInt((ReduceWidth/16 + 1).W))
        if(i != 0){
            prev := mask(i - 1)
        }else{
            prev := io.in.part1
        }
        curr := io.in.part2(i)
        prevandcurr := (prev & curr)
        mask(i) := Mux(prevandcurr === 0.U, prev, prevandcurr)
    }

    val res = Wire(Vec(9 - layers, UInt(1.W)))
    for(i <- 0 until (9 - layers)){
        val bit = Wire(UInt((ReduceWidth/16 + 1).W))
        bit := mask(8 - layers) & io.in.part2(i)
        res(i) := Mux(bit === 0.U, 0.U, 1.U)
    }

    val partres = Wire(UInt((9 - layers).W))
    partres := Cat((0 until (9 - layers)).map(k => res(k)))

  // 生成 out: 9位
    io.out := Cat(io.in.part3, partres)
}

class FPipe0Result(implicit p: Parameters) extends CuteBundle{
    val Product0 = Vec(ReduceWidth/16, SInt(23.W))
    val Product1 = Vec(ReduceWidth/16, SInt(17.W))
    val CMantissa = SInt(32.W)
    val ExceptionVec = Vec(ReduceWidth/16 + 1, new RawFloatException) //每个向量的异常标志位
    val MulExpVec = Vec(ReduceWidth/16 + 1, UInt(9.W))
    val CmpTreeP0Result = new CmpTreeP0Res(cmptreelayers)
    val SignVec = Vec(ReduceWidth/16 + 1, Bool()) //每个向量的符号位
    val opcode = UInt(3.W)   //0:Int8, 1:FP16, 2:BF16, 3:TF32
}

class FPipe1Result(implicit p: Parameters) extends CuteBundle{
    val Product1 = Vec(ReduceWidth/16, SInt(17.W))
    val CMantissa = SInt(32.W)
    val Product0 = Vec(ReduceWidth/16 + 1, SInt(26.W)) //每个向量的尾数右移结果
    val MaxExp = UInt(9.W)
    val opcode = UInt(2.W)  //0:Int8, 1:FP16, 2:BF16, 3:TF32
    val SumException = UInt(4.W) //归约计算结果的异常标志位
    val SignVec = Vec(ReduceWidth/16 + 1, Bool()) //每个向量的符号位
}

class FPipe2Result(implicit p: Parameters) extends CuteBundle{
    val ReduceRes0 = Vec(P3AddNum, SInt((26 + log2Ceil(P2AddNum)).W))
    val ReduceRes1 = Vec(P3AddNum, SInt((17 + log2Ceil(P2AddNum)).W)) // ReduceRes0是尾数右移后的结果，ReduceRes1是Int8的结果
    val CMantissa = SInt(32.W) // C的尾数
    val MaxExp = UInt(9.W)
    val opcode = UInt(2.W)  //0:Int8, 1:FP16, 2:BF16, 3:TF32
    val SumException = UInt(4.W) //归约计算结果的异常标志位
}

class FPipe3Result(implicit p: Parameters) extends CuteBundle{
    val ReduceRes = SInt(32.W) //归约计算结果
    val MaxExp = UInt(9.W)
    val opcode = UInt(2.W)  //0:Int8, 1:FP16, 2:BF16, 3:TF32
    val SumException = UInt(4.W) //归约计算结果的异常标志位
}

// Pipe0的功能是并行完成{计算尾数乘积}和{求阶码最大值，计算右移位数}
class FReduceMACPipe0(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val inA = Input(UInt(ReduceWidth.W))
        val inB = Input(UInt(ReduceWidth.W))
        val inC = Input(UInt(32.W))
        val opcode = Input(UInt(3.W))   //0:Int8, 1:FP16, 2:BF16, 3:TF32, 4:I8 * UI8, 5:UI8 * I8, 6:UI8 * UI8
        val out = Output(new FPipe0Result)
    })

    //输入向量解码
    val ADecoder = Module(new FVecDecoder)
    val BDecoder = Module(new FVecDecoder)

    ADecoder.io.in := io.inA
    ADecoder.io.opcode := Mux(io.opcode === 4.U || io.opcode === 0.U, 0.U, 
        Mux(io.opcode === 5.U || io.opcode === 6.U, 4.U, io.opcode))
    BDecoder.io.in := io.inB
    BDecoder.io.opcode := Mux(io.opcode === 5.U || io.opcode === 0.U, 0.U, 
        Mux(io.opcode === 4.U || io.opcode === 6.U, 4.U, io.opcode))

    //尾数乘法计算

    // Product0
    // 乘法器位宽要求能够满足12位有符号整数的乘法
    for(i <- 0 until ReduceWidth/16){
        val Product0A = Wire(SInt(12.W))
        val Product0B = Wire(SInt(12.W))

        Product0A := Mux(io.opcode === 0.U || io.opcode === 4.U || io.opcode === 5.U || io.opcode === 6.U, ADecoder.io.out.Int8Vec(i).asSInt.pad(12),
                            ADecoder.io.out.TF32Vec(i).signed_sig)
        Product0B := Mux(io.opcode === 0.U || io.opcode === 5.U || io.opcode === 4.U || io.opcode === 6.U, BDecoder.io.out.Int8Vec(i).asSInt.pad(12),
                            BDecoder.io.out.TF32Vec(i).signed_sig)

        io.out.Product0(i) := Product0A * Product0B
        // printf("product0[%d]: %x\n", i.U,io.out.Product0(i))
    }

    // Product1
    // 乘法器位宽要求能够满足9位有符号整数的乘法
    for(i <- ReduceWidth/16 until ReduceWidth/8){
        val Product1A = Wire(SInt(9.W))
        val Product1B = Wire(SInt(9.W))

        Product1A := ADecoder.io.out.Int8Vec(i).asSInt
        Product1B := BDecoder.io.out.Int8Vec(i).asSInt

        io.out.Product1(i - ReduceWidth/16) := Product1A * Product1B
    }

    //阶码计算
    val MulExpVec = Wire(Vec((ReduceWidth/16) + 1, UInt(9.W)))
    val MulExpVecSigned = Wire(Vec((ReduceWidth/16) + 1, SInt(9.W))) //for debug
    for(i <- 0 until ReduceWidth/16){
        MulExpVecSigned(i) := ADecoder.io.out.TF32Vec(i).exp.pad(9) + BDecoder.io.out.TF32Vec(i).exp.pad(9)
        MulExpVec(i) := Mux(ADecoder.io.out.TF32Vec(i).exception.is_zero | BDecoder.io.out.TF32Vec(i).exception.is_zero, 0.U, 
            (ADecoder.io.out.TF32Vec(i).exp.pad(9) + BDecoder.io.out.TF32Vec(i).exp.pad(9)).asUInt + 255.U(9.W))
    }

    for(i <- 0 until ReduceWidth/16){
        val AException = ADecoder.io.out.TF32Vec(i).exception
        val BException = BDecoder.io.out.TF32Vec(i).exception
        io.out.ExceptionVec(i).is_nan := AException.is_nan || BException.is_nan ||
            AException.is_inf && BException.is_zero ||
            BException.is_inf && AException.is_zero

        io.out.ExceptionVec(i).is_zero := AException.is_zero && !BException.is_inf ||
            BException.is_zero && !AException.is_inf

        io.out.ExceptionVec(i).is_inf := !AException.is_zero && BException.is_inf ||
            !BException.is_zero && AException.is_inf

        io.out.SignVec(i) := ADecoder.io.out.TF32Vec(i).sign ^ BDecoder.io.out.TF32Vec(i).sign
    }

    val CDecode = RawFloat.fromUInt(io.inC, 8, 24)
    io.out.SignVec(ReduceWidth/16) := CDecode.sign
    MulExpVec(ReduceWidth/16) := CDecode.exp.pad(9).asUInt + 255.U(9.W) // C的阶码加上偏移量
    MulExpVecSigned(ReduceWidth/16) := CDecode.exp.pad(9) //for debug

    // for(i <- 0 until ReduceWidth/16 + 1){
    //     printf("%x\n", MulExpVecSigned(i).pad(32))
    //     printf("%x\n", MulExpVec(i).pad(32))
    // }

    io.out.ExceptionVec(ReduceWidth/16) := CDecode.exception
    io.out.CMantissa := Mux(io.opcode === 1.U || io.opcode === 2.U || io.opcode === 3.U, 
        CDecode.signed_sig.pad(32), io.inC.asSInt)
    
    // printf("c mantissa: %x\n", io.out.CMantissa)

    // For CmpTree Check
    // val cmptree = Module(new CmpTree)
    // cmptree.io.in := MulExpVec
    // MaxExp := cmptree.io.out
    //取阶码最大值
    val cmptreep0 = Module(new CmpTreeP0(cmptreelayers))
    
    cmptreep0.io.in := MulExpVec
    // printf("%x\n", MaxExp)
    
    // // MaxExpSigned for Debug
    // val MaxExpSigned = Wire(SInt(9.W))
    // MaxExpSigned := (MaxExp - 255.U(9.W)).asSInt
    // printf("%x\n", MaxExpSigned)
    //整理pipe0输出

    //每个乘积的阶码与最大阶码的差值就是右移的位数
    // val RightShiftVec = Wire(Vec(ReduceWidth/16, UInt(9.W)))
    for(i <- 0 until ReduceWidth/16 + 1){
        io.out.MulExpVec(i) := MulExpVec(i)
        // printf("io.out.RightShiftVec[%d]: %x\n", i.U,io.out.RightShiftVec(i))
    }

    //输出opcode和最大阶码
    io.out.opcode := io.opcode
    io.out.CmpTreeP0Result := cmptreep0.io.out
}

// Pipe1的功能是将Pipe0的乘积向量尾数右移，得到尾数向量用于归约计算
class FReduceMACPipe1(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe0Result)
        val out = Output(new FPipe1Result)
    })

    val cmptreep1 = Module(new CmpTreeP1(cmptreelayers))
    cmptreep1.io.in := io.in.CmpTreeP0Result

    val RightShiftVec = Wire(Vec((ReduceWidth/16) + 1, UInt(9.W)))
    for(i <- 0 until ReduceWidth/16 + 1){
        RightShiftVec(i) := cmptreep1.io.out - io.in.MulExpVec(i)
    }

    // 处理exception，下一级流水可以根据HasNaN、HasPInf、HasNInf、OnlyNZero来判断结果
    val NaNVec = Wire(UInt((ReduceWidth/16 + 1).W))
    val PInfVec = Wire(UInt((ReduceWidth/16 + 1).W))
    val NInfVec = Wire(UInt((ReduceWidth/16 + 1).W))
    val NZeroVec = Wire(UInt((ReduceWidth/16 + 1).W))

    NaNVec := Cat((0 until ReduceWidth/16 + 1).map(i => io.in.ExceptionVec(i).is_nan))
    PInfVec := Cat((0 until ReduceWidth/16 + 1).map(i => io.in.ExceptionVec(i).is_inf && !io.in.SignVec(i)))
    NInfVec := Cat((0 until ReduceWidth/16 + 1).map(i => io.in.ExceptionVec(i).is_inf && io.in.SignVec(i)))
    NZeroVec := Cat((0 until ReduceWidth/16 + 1).map(i => io.in.ExceptionVec(i).is_zero && io.in.SignVec(i)))

    val HasNaN = NaNVec.orR
    val HasPInf = PInfVec.orR
    val HasNInf = NInfVec.orR
    val OnlyNZero = NZeroVec.andR

    io.out.SumException := Cat(HasNaN, HasPInf, HasNInf, OnlyNZero)
    //尾数右移
    val ProductRShift = Wire(Vec(ReduceWidth/16 + 1, SInt(26.W)))

    // ProductRShift(ReduceWidth/16) := Mux(io.in.CMantissa < 0.S, (-((- io.in.CMantissa(25, 0).asSInt) >> RightShiftVec(ReduceWidth/16))),
    //     (io.in.CMantissa(25, 0).asSInt >> RightShiftVec(ReduceWidth/16)))

    ProductRShift(ReduceWidth/16) := io.in.CMantissa(25, 0).asSInt >> RightShiftVec(ReduceWidth/16)

    //这里的问题是，如果c是负数，c为补码表示，右移完之后会变成全F，取负再相加后会变成1.与右移完之后变成0的情况不符
    // printf("ProductRShift_C: %x\n", ProductRShift(ReduceWidth/16))
    
    for(i <- 0 until ReduceWidth/16){
        val tempProduct0 = Wire(SInt(26.W))
        tempProduct0 := Cat(io.in.Product0(i)(22, 0), 0.U(3.W)).asSInt
        // ProductRShift(i) := Mux(tempProduct0 < 0.S, -((-tempProduct0) >> RightShiftVec(i)), ((tempProduct0) >> RightShiftVec(i)))
        ProductRShift(i) := tempProduct0 >> RightShiftVec(i)
    }

    for(i <- 0 until ReduceWidth/16 + 1){
        if(i != ReduceWidth/16){
            io.out.Product0(i) := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U, 
                io.in.Product0(i).pad(26), ProductRShift(i))
        } else {
            io.out.Product0(i) := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U, 
                0.S(26.W), ProductRShift(i))
        }
    }

    io.out.CMantissa := io.in.CMantissa
    io.out.Product1 := io.in.Product1
    io.out.MaxExp := cmptreep1.io.out
    io.out.opcode := io.in.opcode
    io.out.SignVec := io.in.SignVec
}

class FReduceMACPipe2(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe1Result)
        val out = Output(new FPipe2Result)
    })

    val ProductRShift = Wire(Vec(ReduceWidth/16, SInt((26 + log2Ceil(P2AddNum)).W)))
    val Int8Product = Wire(Vec(ReduceWidth/16, SInt((17 + log2Ceil(P2AddNum)).W)))

    for(i <- 0 until ReduceWidth/16){
        // ProductRShift(i) := io.in.Product0(i).pad(26 + log2Ceil(P2AddNum))
        ProductRShift(i) := Mux(io.in.SignVec(i), -io.in.Product0(i), io.in.Product0(i)).pad(26 + log2Ceil(P2AddNum))
    }

    for(i <- 0 until ReduceWidth/16){
        Int8Product(i) := io.in.Product1(i).pad(17 + log2Ceil(P2AddNum))
    }


    val SumResult0 = Wire(Vec(P3AddNum, SInt((26 + log2Ceil(P2AddNum)).W)))
    val SumResult1 = Wire(Vec(P3AddNum, SInt((17 + log2Ceil(P2AddNum)).W)))
    // val SumResult0 = Wire(Vec(4, SInt(32.W)))
    // val SumResult1 = Wire(Vec(4, SInt(32.W)))
    for(i <- 0 until P3AddNum) {
        SumResult0(i) := ProductRShift.slice(i * P2AddNum, (i + 1) * P2AddNum).reduce(_ + _)
        SumResult1(i) := Int8Product.slice(i * P2AddNum, (i + 1) * P2AddNum).reduce(_ + _)
    }

    io.out.ReduceRes0 := SumResult0
    io.out.ReduceRes1 := SumResult1
    io.out.CMantissa := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U, 
        io.in.CMantissa, Mux(io.in.SignVec(ReduceWidth/16), -io.in.Product0(ReduceWidth/16), io.in.Product0(ReduceWidth/16)).pad(32))
    io.out.MaxExp := io.in.MaxExp
    io.out.opcode := io.in.opcode
    io.out.SumException := io.in.SumException
}

class FReduceMACPipe3(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe2Result)
        val out = Output(new FPipe3Result)
    })

    val ReduceResF = Wire(SInt(32.W))
    val ReduceResI = Wire(SInt(32.W))
    val ReduceRes1 = Wire(SInt(32.W))
    val SumResult0 = Wire(Vec(P3AddNum, SInt((26 + log2Ceil(ReduceWidth/16)).W)))
    val SumResult1 = Wire(Vec(P3AddNum, SInt((17 + log2Ceil(ReduceWidth/16)).W)))
    for(i <- 0 until P3AddNum) {
        SumResult0(i) := io.in.ReduceRes0(i).pad(26 + log2Ceil(ReduceWidth/16))
        SumResult1(i) := io.in.ReduceRes1(i).pad(17 + log2Ceil(ReduceWidth/16))
    }

    ReduceResF := SumResult0.reduce(_ + _).pad(32) + io.in.CMantissa
    ReduceRes1 := SumResult1.reduce(_ + _).pad(32)
    ReduceResI := ReduceResF + ReduceRes1

    io.out.ReduceRes := Mux(io.in.opcode === 0.U || io.in.opcode === 4.U || io.in.opcode === 5.U || io.in.opcode === 6.U, 
        ReduceResI, ReduceResF)
    io.out.MaxExp := io.in.MaxExp
    io.out.opcode := io.in.opcode
    io.out.SumException := io.in.SumException
}

class FReduceMACPipe4(implicit p: Parameters) extends CuteModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe3Result)
        val out = UInt(ResultWidth.W)
    })

    val ReduceResF = Wire(SInt(32.W))
    val ReduceResI = Wire(SInt(32.W))
    ReduceResF := io.in.ReduceRes
    ReduceResI := io.in.ReduceRes

    val ExceptionBits = Wire(UInt(32.W))
    val IsException = Wire(Bool())
    IsException := io.in.SumException.orR

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
    // printf("UnsignedSig:%x\n", UnsignedSig)

    val clz = Module(new CLZ(32))
    clz.io.in := UnsignedSig
    val LeadingZeros = Wire(UInt(5.W))
    LeadingZeros := clz.io.out
    // printf("LZ:%d\n", LeadingZeros)
    //左移相同，该怎么左移就怎么左移,注意要把最高位的1省略掉作为规格化结果

    //修改一下这里规格化的逻辑
    //真正问题在乘法，乘法尾数是0，但阶码正常计算了
    //最终导致阶码比大小的时候，阶码较大的数尾数是0，不参与加法，而阶码第二大的数，尾数又要右移，导致丢精度，例如fp16_xtest.txt中的数据
    val ShiftMantissa = Wire(SInt(32.W))
    val ShiftExp = Wire(SInt(16.W))

    ShiftExp := (io.in.MaxExp - 255.U(9.W)).asSInt.pad(16) - LeadingZeros.pad(16).asSInt + 8.S(16.W)
    ShiftMantissa := Mux(LeadingZeros > 8.U, UnsignedSig.asSInt << (LeadingZeros - 8.U),
        UnsignedSig.asSInt >> (8.U - LeadingZeros) //这里右移导致移掉了低位的非零位，丢失信息，需要改
    )

    // printf("ShiftMantissa:%x\n", ShiftMantissa)

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
        val DResult = DecoupledIO(UInt(ResultWidth.W))
        val opcode = Input(UInt(3.W))   //0:Int8, 1:FP16, 2:BF16, 3:TF32
    })

    val pipe0 = Module(new FReduceMACPipe0)
    val pipe1 = Module(new FReduceMACPipe1)
    val pipe2 = Module(new FReduceMACPipe2)
    val pipe3 = Module(new FReduceMACPipe3)
    val pipe4 = Module(new FReduceMACPipe4)


    val PipeResRegValid = RegInit(VecInit(Seq.fill(6)(false.B)))

    val InputReg = Reg(UInt((ReduceWidth + ReduceWidth + ResultWidth + 3).W))
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
    io.CAdd.ready := InReady

    val ABCValid = Wire(UInt(1.W))
    ABCValid := io.AVector.valid && io.BVector.valid && io.CAdd.valid

    when(InReady){
        when(ABCValid === 1.U){
            InputReg := Cat(io.AVector.bits, io.BVector.bits, io.CAdd.bits, io.opcode)
            PipeResRegValid(0) := true.B
        }.otherwise{
            InputReg := InputReg
            PipeResRegValid(0) := false.B
        }
    }.otherwise{
        InputReg := InputReg
        PipeResRegValid(0) := PipeResRegValid(0)
    }
    
    pipe0.io.inA := InputReg(ResultWidth + ReduceWidth + ReduceWidth + 2, ResultWidth + ReduceWidth + 3)
    pipe0.io.inB := InputReg(ResultWidth + ReduceWidth + 2, ResultWidth + 3)
    pipe0.io.inC := InputReg(ResultWidth + 2, 3)
    pipe0.io.opcode := InputReg(2, 0)

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

