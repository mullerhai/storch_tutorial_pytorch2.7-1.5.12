package main

import java.io._
import java.net.URL
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import scala.collection.mutable.ArrayBuffer
import scala.util.Using
import org.platanios.torch.api._

class CIFAR100Dataset(root: String, train: Boolean) extends data.Dataset[(Tensor, Tensor)] {
  private val baseUrl = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
  private val filename = "cifar-100-python.tar.gz"
  private val tarPath = s"$root/$filename"
  private val dataDir = s"$root/cifar-100-python"

  // 下载并解压数据
  private def downloadAndExtract(): Unit = {
    if (!new File(dataDir).exists()) {
      if (!new File(tarPath).exists()) {
        println("Downloading CIFAR100 dataset...")
        Using.resource(new BufferedOutputStream(new FileOutputStream(tarPath))) { out =>
          Using.resource(new URL(baseUrl).openStream()) { in =>
            in.transferTo(out)
          }
        }
      }
      println("Extracting CIFAR100 dataset...")
      extractTarGz(tarPath, root)
    }
  }

  // 解压 tar.gz 文件
  private def extractTarGz(tarGzPath: String, destDir: String): Unit = {
    Using.resource(new FileInputStream(tarGzPath)) { fis =>
      Using.resource(new java.util.zip.GZIPInputStream(fis)) { gis =>
        Using.resource(new org.apache.commons.compress.archivers.tar.TarArchiveInputStream(gis)) { tis =>
          var entry: org.apache.commons.compress.archivers.tar.TarArchiveEntry = null
          while ({ entry = tis.getNextTarEntry(); entry != null }) {
            val file = new File(destDir, entry.getName)
            if (entry.isDirectory) {
              file.mkdirs()
            } else {
              file.getParentFile.mkdirs()
              Using.resource(new FileOutputStream(file)) { fos =>
                tis.transferTo(fos)
              }
            }
          }
        }
      }
    }
  }

  private def unpickle(filePath: String): Map[String, Any] = {
    import java.io._
    Using.resource(new FileInputStream(filePath)) { fis =>
      Using.resource(new ObjectInputStream(fis) {
        override def resolveClass(desc: ObjectStreamClass): Class[_] = {
          if (desc.getName == "numpy.ndarray") {
            Class.forName("scala.Array")
          } else {
            super.resolveClass(desc)
          }
        }
      }) { ois =>
        ois.readObject().asInstanceOf[Map[String, Any]]
      }
    }
  }

  private val (images, labels) = {
    downloadAndExtract()
    val dataFile = if (train) s"$dataDir/train" else s"$dataDir/test"
    val unpickledData = unpickle(dataFile)
    val rawImages = unpickledData("data").asInstanceOf[Array[Array[Byte]]]
    val rawLabels = unpickledData("fine_labels").asInstanceOf[List[Int]]

    val imageTensors = rawImages.map { img =>
      val reshaped = img.grouped(1024).toArray
      val stacked = Tensor.stack(
        reshaped.map(channel => Tensor.fromArray(channel.map(_.toFloat / 255.0f)).reshape(32, 32))
      )
      stacked
    }
    (imageTensors, rawLabels.toArray.map(Tensor.fromInt))
  }

  override def size: Int = images.length

  override def apply(index: Int): (Tensor, Tensor) = {
    (images(index), labels(index))
  }
}

object CIFAR100Dataset {
  def main(args: Array[String]): Unit = {
    val dataset = new CIFAR100Dataset(root = "./data", train = true)
    println(s"Dataset size: ${dataset.size}")
    val (image, label) = dataset(0)
    println(s"Image shape: ${image.shape}")
    println(s"Label: ${label.item[Int]}")
  }
}
