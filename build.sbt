ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.7.1"

lazy val root = (project in file("."))
  .settings(
    name := "scala3test"
  )

libraryDependencies += "io.github.mullerhai" % "storch_core_3" % "0.3.7-1.15.2"
libraryDependencies += "org.bytedeco" % "pytorch" % "2.7.1-1.5.12"
libraryDependencies += "junit" % "junit" % "4.13.2" % Test
libraryDependencies += "org.scalameta" %% "munit" % "1.1.1"  % Test
libraryDependencies += "org.scalameta" %% "munit-scalacheck" % "1.1.0" % Test //"0.7.29"
//libraryDependencies += "io.github.mullerhai" % "core_3" %   "0.2.6-1.15.1"
//libraryDependencies += "io.github.mullerhai" % "storch_vision_3" % "0.3.0-1.15.1"
//libraryDependencies +=   "dev.storch" % "vision_3" % "0.2.3-1.15.1"
// https://mvnrepository.com/artifact/org.bytedeco/openblas-platform
//libraryDependencies += "org.bytedeco" % "openblas-platform" % "0.3.28-1.5.11"
//// https://mvnrepository.com/artifact/org.bytedeco/mkl-platform
//libraryDependencies += "org.bytedeco" % "mkl-platform" % "2025.0-1.5.11"
