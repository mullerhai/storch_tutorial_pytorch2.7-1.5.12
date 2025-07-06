//package basic
//
//import jdk.incubator.vector.{VectorShape, VectorSpecies}
//
//import java.lang.foreign.*
//import java.lang.foreign.MemoryLayout.PathElement.*
//import java.lang.invoke.*
//import java.util.stream.*
//
//object MatrixOperations {
//  private val BIT_SET = 256
//  private val vectorSpecies = VectorSpecies.of(java.lang.Float.TYPE, VectorShape.forBitSize(BIT_SET))
//
//  def matrixAddition(a: Array[Array[Float]], b: Array[Array[Float]]): Array[Array[Float]] = {
//    require(a.length == b.length && a(0).length == b(0).length, "矩阵维度必须一致")
//    val result = Array.ofDim[Float](a.length, a(0).length)
//    for (i <- a.indices) {
//      var j = 0
//      while (j < a(i).length) {
//        val upperBound = Math.min(j + vectorSpecies.length(), a(i).length)
//        val va = vectorSpecies.fromArray(a(i), j)
//        val vb = vectorSpecies.fromArray(b(i), j)
//        val vc = va.add(vb)
//        vc.intoArray(result(i), j)
//        j = upperBound
//      }
//    }
//    result
//  }
//
//  def matrixSubtraction(a: Array[Array[Float]], b: Array[Array[Float]]): Array[Array[Float]] = {
//    require(a.length == b.length && a(0).length == b(0).length, "矩阵维度必须一致")
//    val result = Array.ofDim[Float](a.length, a(0).length)
//    for (i <- a.indices) {
//      var j = 0
//      while (j < a(i).length) {
//        val upperBound = Math.min(j + vectorSpecies.length(), a(i).length)
//        val va = vectorSpecies.fromArray(a(i), j)
//        val vb = vectorSpecies.fromArray(b(i), j)
//        val vc = va.sub(vb)
//        vc.intoArray(result(i), j)
//        j = upperBound
//      }
//    }
//    result
//  }
//
//  def matrixMultiplication(a: Array[Array[Float]], b: Array[Array[Float]]): Array[Array[Float]] = {
//    require(a(0).length == b.length, "第一个矩阵的列数必须等于第二个矩阵的行数")
//    val result = Array.ofDim[Float](a.length, b(0).length)
//    for (i <- a.indices; k <- b.indices; j <- b(0).indices) {
//      result(i)(j) += a(i)(k) * b(k)(j)
//    }
//    result
//  }
//
//  def matrixDivision(a: Array[Array[Float]], b: Array[Array[Float]]): Array[Array[Float]] = {
//    require(a.length == b.length && a(0).length == b(0).length, "矩阵维度必须一致")
//    val result = Array.ofDim[Float](a.length, a(0).length)
//    for (i <- a.indices) {
//      var j = 0
//      while (j < a(i).length) {
//        val upperBound = Math.min(j + vectorSpecies.length(), a(i).length)
//        val va = vectorSpecies.fromArray(a(i), j)
//        val vb = vectorSpecies.fromArray(b(i), j)
//        val vc = va.div(vb)
//        vc.intoArray(result(i), j)
//        j = upperBound
//      }
//    }
//    result
//  }
//
//  def main(args: Array[String]): Unit = {
//    val a = Array(
//      Array(1.0f, 2.0f),
//      Array(3.0f, 4.0f)
//    )
//    val b = Array(
//      Array(5.0f, 6.0f),
//      Array(7.0f, 8.0f)
//    )
//
//    println("矩阵加法结果:")
//    matrixAddition(a, b).foreach(row => println(row.mkString(" ")))
//
//    println("矩阵减法结果:")
//    matrixSubtraction(a, b).foreach(row => println(row.mkString(" ")))
//
//    println("矩阵乘法结果:")
//    matrixMultiplication(a, b).foreach(row => println(row.mkString(" ")))
//
//    println("矩阵除法结果:")
//    matrixDivision(a, b).foreach(row => println(row.mkString(" ")))
//  }
//}
