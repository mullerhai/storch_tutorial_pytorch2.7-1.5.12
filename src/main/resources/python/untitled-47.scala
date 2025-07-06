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
import java.util.zip.ZipInputStream

class ImdbDataset(root: String, train: Boolean = true, download: Boolean = true) extends Dataset[(Tensor, Tensor)] {
  private val baseUrl = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
  private val tarGzPath = Paths.get(root, "aclImdb_v1.tar.gz")
  private val dataDir = Paths.get(root, "aclImdb")
  private val trainDir = Paths.get(dataDir.toString, "train")
  private val testDir = Paths.get(dataDir.toString, "test")

  if (download) {
    downloadData()
  }

  private val dataDirToUse = if (train) trainDir else testDir
  if (!Files.exists(dataDirToUse)) {
    throw new RuntimeException("Dataset not found. You can use download = true to download it.")
  }

  private var reviews: Seq[Tensor] = _
  private var labels: Seq[Tensor] = _
  private val vocab = mutable.Map[String, Int]().withDefaultValue(0)
  private val maxVocabSize = 10000

  loadData()

  private def downloadData(): Unit = {
    if (Files.exists(dataDir)) return
    Files.createDirectories(Paths.get(root))
    println("Downloading IMDB dataset...")
    val in = new URL(baseUrl).openStream()
    val out = Files.newOutputStream(tarGzPath)
    in.transferTo(out)
    in.close()
    out.close()
    println("Download complete. Extracting...")
    import sys.process._
    s"tar -xzf ${tarGzPath} -C ${root}".!
    println("Extraction complete.")
  }

  private def buildVocab(): Unit = {
    val wordCount = mutable.Map[String, Int]()
    val dirs = Seq(Paths.get(trainDir.toString, "pos"), Paths.get(trainDir.toString, "neg"))
    for {
      dir <- dirs
      file <- Files.newDirectoryStream(dir)
      content = Files.readString(file)
      words = content.toLowerCase().split("\\W+")
      word <- words
    } {
      wordCount(word) = wordCount.getOrElse(word, 0) + 1
    }

    val sortedWords = wordCount.toSeq.sortBy(-_._2).take(maxVocabSize - 1)
    sortedWords.zipWithIndex.foreach { case ((word, _), idx) =>
      vocab(word) = idx + 1
    }
  }

  private def loadData(): Unit = {
    buildVocab()
    println("Loading IMDB dataset...")
    val allReviews = mutable.ListBuffer[Tensor]()
    val allLabels = mutable.ListBuffer[Tensor]()

    val posDir = Paths.get(dataDirToUse.toString, "pos")
    val negDir = Paths.get(dataDirToUse.toString, "neg")

    for {
      (dir, label) <- Seq((posDir, 1), (negDir, 0))
      file <- Files.newDirectoryStream(dir)
    } {
      val content = Files.readString(file)
      val words = content.toLowerCase().split("\\W+")
      val reviewIndices = words.map(word => vocab(word)).toArray
      allReviews += Tensor(reviewIndices, dtype = Int64)
      allLabels += Tensor(label, dtype = Float32)
    }

    reviews = allReviews.toSeq
    labels = allLabels.toSeq
  }

  override def size: Int = reviews.size

  override def apply(index: Int): (Tensor, Tensor) = {
    (reviews(index), labels(index))
  }
}

object ImdbDatasetExample extends App {
  // 初始化训练集
  val trainDataset = new ImdbDataset(root = "./data", train = true, download = true)
  // 初始化测试集
  val testDataset = new ImdbDataset(root = "./data", train = false)

  println(s"Train dataset size: ${trainDataset.size}")
  println(s"Test dataset size: ${testDataset.size}")

  // 获取第一个样本
  val (review, label) = trainDataset(0)
  println(s"Review indices length: ${review.size}, Label: ${label}")

  // 使用 DataLoader
  val trainLoader = DataLoader(trainDataset, batchSize = 16, shuffle = true)
  trainLoader.foreach { batch =>
    val (batchReviews, batchLabels) = batch
    println(s"Batch - Reviews shape: ${batchReviews.shape}, Labels shape: ${batchLabels.shape}")
  }
}
