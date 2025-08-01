package cute
// package cute

import chisel3._
import chisel3.util._
// import org.chipsalliance.cde.config._
// // import boom.exu.ygjk._
// import boom.v3.util._



trait CUTEImplParameters{
  val ReduceWidth :Int = 512
  val ResultWidth :Int = 32
  val cmptreelayers :Int = 4
  val P3AddNum :Int = 4
  val P2AddNum :Int = ReduceWidth / (P3AddNum * 16)
}

class Parameters{}

class CuteModule(implicit val p: Parameters) extends Module with CUTEImplParameters
class CuteBundle(implicit val p: Parameters) extends Bundle with CUTEImplParameters