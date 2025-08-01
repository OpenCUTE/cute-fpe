package cute

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.BundleLiterals._
import scala.io.StdIn
import java.io._

// Let's now generate modules with different widths
object Main extends App {
  // (new ChiselStage).emitVerilog(new multiplier(12,8))
  val writer = new PrintWriter(new File("top.v"))  //specify the file path
  writer.write(getVerilogString(new top()(new Parameters())))
  writer.close()
  // println(getVerilogString(new multiplier(12, 8)))
}