pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()

        // Meta Wearables Device Access Toolkit (MWDAT)
        // Requires a GitHub PAT with read:packages scope
        // Set GITHUB_TOKEN env var or github_token in local.properties
        maven {
            url = uri("https://maven.pkg.github.com/facebook/meta-wearables-dat-android")
            credentials {
                val localProps = java.util.Properties()
                val localPropsFile = rootDir.resolve("local.properties")
                if (localPropsFile.exists()) {
                    localProps.load(localPropsFile.inputStream())
                }
                
                username = "devon" // GitHub packages usually requires actual username for PATs
                password = localProps.getProperty("github_token")
                    ?: providers.gradleProperty("github_token").orNull
                    ?: System.getenv("GITHUB_TOKEN")
                    ?: ""
            }
        }
    }
}

rootProject.name = "Less"
include(":app")
