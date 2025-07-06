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

class CriteoDataset(root: String, train: Boolean = true, download: Boolean = false) extends Dataset[(Tensor, Tensor, Tensor)] {
  private val trainUrl = "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz"
  private val trainPath = Paths.get(root, "dac.tar.gz")
  private val dataDir = Paths.get(root, "train.txt")

  if (download) {
    downloadData()
  }

  if (!Files.exists(dataDir)) {
    throw new RuntimeException("Dataset not found. You can use download = true to download it.")
  }

  private var labels: Seq[Tensor] = _
  private var numericalFeatures: Seq[Tensor] = _
  private var categoricalFeatures: Seq[Tensor] = _

  loadData()
  splitData()

  private def downloadData(): Unit = {
    if (Files.exists(dataDir)) return
    Files.createDirectories(Paths.get(root))
    println("Downloading Criteo dataset...")
    // 注意：实际下载可能需要处理网络请求和认证
    val in = new URL(trainUrl).openStream()
    val out = Files.newOutputStream(trainPath)
    in.transferTo(out)
    in.close()
    out.close()
    println("Download complete. Extracting...")
    // 解压 tar.gz 文件
    import sys.process._
    s"tar -xzf ${trainPath} -C ${root}".!
    println("Extraction complete.")
  }

  private def loadData(): Unit = {
    println("Loading Criteo dataset...")
    val allLabels = mutable.ListBuffer[Tensor]()
    val allNumericalFeatures = mutable.ListBuffer[Tensor]()
    val allCategoricalFeatures = mutable.ListBuffer[Tensor]()

    Using.resource(Files.newBufferedReader(dataDir)) { reader =>
      var line = reader.readLine()
      while (line != null) {
        val parts = line.split('\t')
        val label = parts(0).toFloat
        val numericalPart = parts.slice(1, 14)
        val categoricalPart = parts.slice(14, parts.length)

        val numericalValues = numericalPart.map { value =>
          if (value.isEmpty) 0.0f else value.toFloat
        }
        val categoricalValues = categoricalPart.map(_.hashCode().toLong)

        allLabels += Tensor(label, dtype = Float32)
        allNumericalFeatures += Tensor(numericalValues, dtype = Float32)
        allCategoricalFeatures += Tensor(categoricalValues, dtype = Int64)

        line = reader.readLine()
      }
    }

    labels = allLabels.toSeq
    numericalFeatures = allNumericalFeatures.toSeq
    categoricalFeatures = allCategoricalFeatures.toSeq
  }

  private def splitData(): Unit = {
    val trainSize = (labels.size * 0.8).toInt
    if (train) {
      labels = labels.take(trainSize)
      numericalFeatures = numericalFeatures.take(trainSize)
      categoricalFeatures = categoricalFeatures.take(trainSize)
    } else {
      labels = labels.drop(trainSize)
      numericalFeatures = numericalFeatures.drop(trainSize)
      categoricalFeatures = categoricalFeatures.drop(trainSize)
    }
  }

  override def size: Int = labels.size

  override def apply(index: Int): (Tensor, Tensor, Tensor) = {
    (labels(index), numericalFeatures(index), categoricalFeatures(index))
  }
}

object CriteoDatasetExample extends App {
  // 初始化训练集
  val trainDataset = new CriteoDataset(root = "./data", train = true, download = true)
  // 初始化测试集
  val testDataset = new CriteoDataset(root = "./data", train = false)

  println(s"Train dataset size: ${trainDataset.size}")
  println(s"Test dataset size: ${testDataset.size}")

  // 获取第一个样本
  val (label, numericalFeature, categoricalFeature) = trainDataset(0)
  println(s"Label: ${label}, Numerical Feature shape: ${numericalFeature.shape}, Categorical Feature shape: ${categoricalFeature.shape}")

  // 使用 DataLoader
  val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true)
  trainLoader.foreach { batch =>
    val (batchLabels, batchNumericalFeatures, batchCategoricalFeatures) = batch
    println(s"Batch - Labels shape: ${batchLabels.shape}, Numerical Features shape: ${batchNumericalFeatures.shape}, Categorical Features shape: ${batchCategoricalFeatures.shape}")
  }
}
