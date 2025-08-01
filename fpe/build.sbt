name := "cute"

// // crossScalaVersions := Seq("2.12.14", "2.13.10")
import Tests._

val chisel6Version = "6.7.0"
val chiselTestVersion = "6.0.0"
val scalaVersionFromChisel = "2.13.16"

val chisel3Version = "3.6.1"

// This gives us a nicer handle to the root project instead of using the
// implicit one
// lazy val chipyardRoot = Project("chipyardRoot", file("."))

scalaVersion := scalaVersionFromChisel
scalacOptions ++= Seq(
  "-deprecation",
  "-unchecked",
  "-Ytasty-reader",
  "-Ymacro-annotations")// fix hierarchy API
// unmanagedBase := (chipyardRoot / unmanagedBase).value,
// allDependencies := {
//   // drop specific maven dependencies in subprojects in favor of Chipyard's version
//   val dropDeps = Seq(("edu.berkeley.cs", "rocketchip"))
//   allDependencies.value.filterNot { dep =>
//    dropDeps.contains((dep.organization, dep.name))
//   }
// }
libraryDependencies += "com.lihaoyi" %% "sourcecode" % "0.3.1"
libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value

exportJars := true
resolvers ++= Seq(
  Resolver.sonatypeRepo("snapshots"),
  Resolver.sonatypeRepo("releases"),
  Resolver.mavenLocal)

// scalacOptions ++= Seq(
//   "-deprecation",
//   "-feature",
//   "-unchecked",
//   "-Xfatal-warnings",
//   "-language:reflectiveCalls",
// )

libraryDependencies ++= Seq("org.chipsalliance" %% "chisel" % chisel6Version)
addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chisel6Version cross CrossVersion.full)


libraryDependencies ++= Seq(
  "org.apache.commons" % "commons-lang3" % "3.12.0",
  "org.apache.commons" % "commons-text" % "1.9"
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "3.2.+" % "test"
)

libraryDependencies ++= Seq(
  "com.lihaoyi" %% "mainargs" % "0.5.0",
  "org.json4s" %% "json4s-jackson" % "4.0.5",
  "org.scala-graph" %% "graph-core" % "1.13.5"
)

// addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % "3.5.6" cross CrossVersion.full)
// libraryDependencies += "edu.berkeley.cs" %% "chisel3" % "3.5.6"
// libraryDependencies += "edu.berkeley.cs" %% "chiseltest" % "0.5.6"