//  def matrixAddition(a: Array[Array[Float]], b: Array[Array[Float]]): Array[Array[Float]] = {
//    println(s"a: ${a.mkString(",")} b ${b.mkString(",")} vectorSpecies ${vectorSpecies}")
//    require(a.length == b.length && a(0).length == b(0).length, "矩阵维度必须一致")
//    val result = Array.ofDim[Float](a.length, a(0).length)
//    for (i <- a.indices) {
//      var j = 0
//      while (j < a(i).length) {
//        val upperBound = Math.min(j + vectorSpecies.length(), a(i).length)
//        val copyLength = upperBound - j
//        if (copyLength == vectorSpecies.length()) {
//          // 当剩余元素数量等于向量长度时，使用向量操作
//          val va = vectorSpecies.fromArray(a(i), j)
//          val vb = vectorSpecies.fromArray(b(i), j)
//          val vc = va.add(vb)
//          val vcArray = vc.toArray
//          Array.copy(vcArray, 0, result(i), j, copyLength)
//        } else {
//          // 当剩余元素数量小于向量长度时，使用标量操作
//          for (k <- 0 until copyLength) {
//            result(i)(j + k) = a(i)(j + k) + b(i)(j + k)
//          }
//        }
//        j = upperBound
//      }
//    }
//    result
//  }
