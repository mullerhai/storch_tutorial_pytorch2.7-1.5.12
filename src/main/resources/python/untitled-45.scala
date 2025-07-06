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

class MovielensDataset(root: String, train: Boolean = true, download: Boolean = true) extends Dataset[(Tensor, Tensor, Tensor)] {
  private val baseUrl = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
  private val zipPath = Paths.get(root, "ml-100k.zip")
  private val dataDir = Paths.get(root, "ml-100k")
  private val dataPath = Paths.get(dataDir.toString, "u.data")

  if (download) {
    downloadData()
  }

  if (!Files.exists(dataPath)) {
    throw new RuntimeException("Dataset not found. You can use download = true to download it.")
  }

  private var userIds: Seq[Tensor] = _
  private var movieIds: Seq[Tensor] = _
  private var ratings: Seq[Tensor] = _

  loadData()
  splitData()

  private def downloadData(): Unit = {
    if (Files.exists(dataPath)) return
    Files.createDirectories(Paths.get(root))
    println("Downloading MovieLens dataset...")
    val in = new URL(baseUrl).openStream()
    val out = Files.newOutputStream(zipPath)
    in.transferTo(out)
    in.close()
    out.close()
    println("Download complete. Extracting...")
    import java.util.zip._
    Using.resource(new ZipInputStream(Files.newInputStream(zipPath))) { zis =>
      var entry = zis.getNextEntry
      while (entry != null) {
        val filePath = Paths.get(root, entry.getName)
        if (!entry.isDirectory) {
          Files.createDirectories(filePath.getParent)
          Using.resource(Files.newOutputStream(filePath)) { fos =>
            zis.transferTo(fos)
          }
        }
        zis.closeEntry()
        entry = zis.getNextEntry
      }
    }
    println("Extraction complete.")
  }

  private def loadData(): Unit = {
    println("Loading MovieLens dataset...")
    val allUserIds = mutable.ListBuffer[Tensor]()
    val allMovieIds = mutable.ListBuffer[Tensor]()
    val allRatings = mutable.ListBuffer[Tensor]()

    Using.resource(Files.newBufferedReader(dataPath)) { reader =>
      var line = reader.readLine()
      while (line != null) {
        val parts = line.split("\t")
        val userId = parts(0).toInt
        val movieId = parts(1).toInt
        val rating = parts(2).toFloat

        allUserIds += Tensor(userId, dtype = Int64)
        allMovieIds += Tensor(movieId, dtype = Int64)
        allRatings += Tensor(rating, dtype = Float32)

        line = reader.readLine()
      }
    }

    userIds = allUserIds.toSeq
    movieIds = allMovieIds.toSeq
    ratings = allRatings.toSeq
  }

  private def splitData(): Unit = {
    val trainSize = (userIds.size * 0.8).toInt
    if (train) {
      userIds = userIds.take(trainSize)
      movieIds = movieIds.take(trainSize)
      ratings = ratings.take(trainSize)
    } else {
      userIds = userIds.drop(trainSize)
      movieIds = movieIds.drop(trainSize)
      ratings = ratings.drop(trainSize)
    }
  }

  override def size: Int = userIds.size

  override def apply(index: Int): (Tensor, Tensor, Tensor) = {
    (userIds(index), movieIds(index), ratings(index))
  }
}

object MovielensDatasetExample extends App {
  // 初始化训练集
  val trainDataset = new MovielensDataset(root = "./data", train = true, download = true)
  // 初始化测试集
  val testDataset = new MovielensDataset(root = "./data", train = false)

  println(s"Train dataset size: ${trainDataset.size}")
  println(s"Test dataset size: ${testDataset.size}")

  // 获取第一个样本
  val (userId, movieId, rating) = trainDataset(0)
  println(s"User ID: ${userId}, Movie ID: ${movieId}, Rating: ${rating}")

  // 使用 DataLoader
  val trainLoader = DataLoader(trainDataset, batchSize = 64, shuffle = true)
  trainLoader.foreach { batch =>
    val (batchUserIds, batchMovieIds, batchRatings) = batch
    println(s"Batch - User IDs shape: ${batchUserIds.shape}, Movie IDs shape: ${batchMovieIds.shape}, Ratings shape: ${batchRatings.shape}")
  }
}
