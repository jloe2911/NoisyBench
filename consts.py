import os

JAVA_HOME_PATH = r"C:\Program Files\Java\jdk-25"
JVM_MEMORY = '10g'

os.environ["JAVA_HOME"] = JAVA_HOME_PATH
os.environ["PATH"] = JAVA_HOME_PATH + r"\bin;" + os.environ["PATH"]
