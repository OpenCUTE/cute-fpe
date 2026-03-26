package cute
// package cute

import chisel3._
import chisel3.util._
// import org.chipsalliance.cde.config._
// // import boom.exu.ygjk._
// import boom.v3.util._
import org.chipsalliance.cde.config._


trait CUTEImplParameters{
  val ReduceWidth :Int = 512
  val ResultWidth :Int = 32
  val MinGroupSize :Int = 16
  val MinDataTypeWidth : Int = 4
  val ScaleElementWidth : Int = 8
  val cmptreelayers :Int = 4
  val fp8cmptreelayers :Int = 4

  // 目前固定，与FP4刚好公用，不要动
  val P3AddNum :Int = ReduceWidth / 4 / MinGroupSize
  val P2AddNum :Int = ReduceWidth / (P3AddNum * 16)
  // val P4AddNum :Int = ReduceWidth / (P3AddNum * P2AddNum * 16)     // 刚好为1

  val FP4P0AddNum :Int = 2
  val FP4P1AddNum :Int = 16 / FP4P0AddNum

  var DEBUG_FP8 :Boolean = true
  var DEBUG_FP4 :Boolean = false
}

class Parameters{}

class CuteModule(implicit val p: Parameters) extends Module with CUTEImplParameters
class CuteBundle(implicit val p: Parameters) extends Bundle with CUTEImplParameters