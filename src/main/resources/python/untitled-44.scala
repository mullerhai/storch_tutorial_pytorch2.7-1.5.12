import java.io.{BufferedReader, InputStreamReader}
import java.net.URL
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream
import scala.collection.mutable
import scala.util.Random
import org.platanios.tensorflow.api._
import org.platanios.torch.jni
import org.platanios.torch.api._
import org.platanios.torch.api.data.{Dataset, DataLoader}
import org.platanios.torch.api.tensors.Tensor
import scala.util.Using

class AvazuDataset(root: String, train: Boolean = true, download: Boolean = false) extends Dataset[(Tensor, Tensor)] {
  private val trainUrl = "https://www.kaggle.com/c/avazu-ctr-prediction/download/train.gz"
  private val trainPath = Paths.get(root, "train.gz")

  if (download) {
    downloadData()
  }

  if (!Files.exists(trainPath)) {
    throw new RuntimeException("Dataset not found. You can use download = true to download it.")
  }

  private var features: Seq[Tensor] = _
  private var labels: Seq[Tensor] = _

  loadData()
  splitData()

  private def downloadData(): Unit = {
    if (Files.exists(trainPath)) return
    Files.createDirectories(Paths.get(root))
    println("Downloading Avazu dataset...")
    // 注意：Kaggle 需要认证，实际下载可能需要使用 Kaggle API 或手动下载
    // 这里只是示例代码，实际运行可能需要修改
    val in = new URL(trainUrl).openStream()
    val out = Files.newOutputStream(trainPath)
    in.transferTo(out)
    in.close()
    out.close()
    println("Download complete.")
  }

  private def loadData(): Unit = {
    println("Loading Avazu dataset ungzip ...")
    val allFeatures = mutable.ListBuffer[Tensor]()
    val allLabels = mutable.ListBuffer[Tensor]()

    Using.resource(new GZIPInputStream(Files.newInputStream(trainPath))) { gis =>
      Using.resource(new BufferedReader(new InputStreamReader(gis))) { reader =>
        val header = reader.readLine().split(',')
        val catColIndices = header.zipWithIndex.collect {
          case (col, idx) if col != "id" && col != "click" && header(idx).forall(_.isLetter) => idx
        }.toSet

        var line = reader.readLine()
        while (line != null) {
          val parts = line.split(',')
          val label = parts(1).toFloat
          val featureValues = mutable.ListBuffer[Float]()

          for (i <- 2 until parts.length) {
            if (catColIndices.contains(i)) {
              // 简单的哈希编码处理分类特征
              featureValues += parts(i).hashCode().toFloat
            } else {
              featureValues += parts(i).toFloat
            }
          }

          allFeatures += Tensor(featureValues.toArray, dtype = Float32)
          allLabels += Tensor(label, dtype = Float32)
          line = reader.readLine()
        }
      }
    }

    features = allFeatures.toSeq
    labels = allLabels.toSeq
  }

  private def splitData(): Unit = {
    val trainSize = (features.size * 0.8).toInt
    if (train) {
      features = features.take(trainSize)
      labels = labels.take(trainSize)
    } else {
      features = features.drop(trainSize)
      labels = labels.drop(trainSize)
    }
  }

  override def size: Int = features.size

  override def apply(index: Int): (Tensor, Tensor) = {
    (features(index), labels(index))
  }
}

object AvazuDatasetExample extends App {
  // 初始化训练集
  val trainDataset = new AvazuDataset(root = "./data", train = true, download = false)
  // 初始化测试集
  val testDataset = new AvazuDataset(root = "./data", train = false)

  println(s"Train dataset size: ${trainDataset.size}")
  println(s"Test dataset size: ${testDataset.size}")

  // 获取第一个样本
  val (features, label) = trainDataset(0)
  println(s"Features: ${features}, Label: ${label}")

  // 使用 DataLoader
  val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true)
  trainLoader.foreach { batch =>
    val (batchFeatures, batchLabels) = batch
    println(s"Batch - Features shape: ${batchFeatures.shape}, Labels shape: ${batchLabels.shape}")
  }
}
