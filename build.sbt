ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.6.4"

lazy val root = (project in file("."))
  .settings(
    name := "scala3test"
  )
Compile / scalacOptions ++= Seq("--add-modules", "jdk.incubator.vector")
run / fork := true
run / javaOptions ++= Seq("--add-modules", "jdk.incubator.vector")
//resolvers += Resolver.sonatypeRepo("releases")
resolvers += "center" at "https://repo1.maven.org/maven2/"
//resolvers += "aliyun" at "https://maven.aliyun.com/repository/public"
libraryDependencies += "io.github.mullerhai" % "storch-numpy_3" % "0.1.1"


libraryDependencies += "io.github.mullerhai" % "storch_core_3" % "0.3.9-1.15.2"
libraryDependencies += "org.bytedeco" % "pytorch" % "2.7.1-1.5.12"
libraryDependencies += "org.bytedeco" % "pytorch-platform-gpu" % "2.7.1-1.5.12"  //classifier "windows-x86_64"
libraryDependencies += "org.bytedeco" % "openblas" % "0.3.30-1.5.12"
libraryDependencies += "org.bytedeco" % "openblas-platform" % "0.3.30-1.5.12"
libraryDependencies += "org.bytedeco" % "cuda" % "12.9-9.10-1.5.12"
libraryDependencies += "org.bytedeco" % "cuda-platform" % "12.9-9.10-1.5.12"
// https://mvnrepository.com/artifact/io.github.manzurola/spacy4j
libraryDependencies += "io.github.manzurola" % "spacy4j" % "0.4.0" pomOnly()
// https://mvnrepository.com/artifact/org.imixs.workflow/imixs-ml-spacy
libraryDependencies += "org.imixs.workflow" % "imixs-ml-spacy" % "1.1.6" pomOnly()
// https://mvnrepository.com/artifact/org.renjin.cran/cleanNLP
//libraryDependencies += "org.renjin.cran" % "cleanNLP" % "2.0.3-b1"
// https://mvnrepository.com/artifact/org.renjin.cran/spacyr
//libraryDependencies += "org.renjin.cran" % "spacyr" % "0.9.9-b1"
// https://mvnrepository.com/artifact/com.github.texttechnologylab.textimager-uima/textimager-uima-spacy
//libraryDependencies += "com.github.texttechnologylab.textimager-uima" % "textimager-uima-spacy" % "0.3.0.2"
// https://mvnrepository.com/artifact/io.github.manzurola/spacy4j-adapters-spacy-server
libraryDependencies += "io.github.manzurola" % "spacy4j-adapters-spacy-server" % "0.4.0"
//libraryDependencies += "org.bytedeco" % "cuda-platform-redist" % "12.6-9.5-1.5.11"
libraryDependencies += "junit" % "junit" % "4.13.2" % Test
libraryDependencies += "org.scalameta" %% "munit" % "1.1.1"  % Test
libraryDependencies += "org.scalameta" %% "munit-scalacheck" % "1.1.0" % Test //"0.7.29
//libraryDependencies += "org.bytedeco" % "mkl" % "2024.1.0-1.5.12"
//libraryDependencies += "org.bytedeco" % "mkl-platform" % "2024.1.0-1.5.12"
// https://mvnrepository.com/artifact/org.bytedeco/cuda
//libraryDependencies += "io.github.mullerhai" % "core_3" %   "0.2.6-1.15.1"
//libraryDependencies += "io.github.mullerhai" % "storch_vision_3" % "0.3.0-1.15.1"
//libraryDependencies +=   "dev.storch" % "vision_3" % "0.2.3-1.15.1"
// https://mvnrepository.com/artifact/org.bytedeco/openblas-platform
//libraryDependencies += "org.bytedeco" % "openblas-platform" % "0.3.28-1.5.11"
//// https://mvnrepository.com/artifact/org.bytedeco/mkl-platform
//libraryDependencies += "org.bytedeco" % "mkl-platform" % "2025.0-1.5.11"
