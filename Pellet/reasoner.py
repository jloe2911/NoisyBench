import sys
if len(sys.argv) < 2:
    raise ValueError("You must provide dataset_name as the first argument")
dataset_name = sys.argv[1]

import subprocess
from os import environ, popen
from pathlib import Path
from sys import executable, platform
from rdflib import Graph

JAVA_EXE = None

if platform == "linux" or platform == "linux2":
    # linux
    JAVA_EXE = "/usr/bin/java"
elif platform == "darwin":
    # OS X
    JAVA_EXE = "/usr/bin/java"
elif platform == "win32":
    # Windows
    JAVA_EXE = "java"

if JAVA_EXE is None:
    raise RuntimeError("Unsupported platform")

# Test the Java version
java_version = popen(f"{JAVA_EXE} -version").read()

JAVA_MEM = environ.get("JAVA_MEM", 2048)

MAX_SUBPROCESS = int(environ.get("MAX_SUBPROCESS", 1))

# Python executable
PYTHON = executable

# Collect all graph files from the data directory
data_dir = Path(f"../datasets/{dataset_name}_input_graphs")
temp_dir = Path("./temp")
result_dir = Path(
    f"../datasets/{dataset_name}_inferred_graphs"
)

if not temp_dir.exists():
    temp_dir.mkdir()
if not result_dir.exists():
    result_dir.mkdir()

# If temp is not empty, remove all files
else:
    for file in temp_dir.glob("*"):
        file.unlink()

if not result_dir.exists():
    result_dir.mkdir(parents=True)

ttl_files = list(data_dir.glob("*.ttl"))
owl_files = list(data_dir.glob("*.owl"))
xml_files = list(data_dir.glob("*.xml"))

# Convert ttl files to xml files and save them in the temp directory
for ttl_file in ttl_files:
    graph = Graph()
    graph.parse(str(ttl_file), format="turtle")
    graph.serialize(
        destination=temp_dir / (ttl_file.stem + ".xml"),
        format="xml",
    )

# Move other files to the temp directory
for owl_file in owl_files:
    owl_file.rename(temp_dir / owl_file.name)

for xml_file in xml_files:
    xml_file.rename(temp_dir / xml_file.name)

CURRENT_RUNNING = []

COMMAND_PIPE = [
    str(PYTHON),
    str(Path("run_reasoner.py").absolute()),
    str(JAVA_EXE),
    str(JAVA_MEM),
]

# For all files in the temp directory, run the reasoner
for file in temp_dir.glob("*"):
    # If we have reached the maximum number of subprocesses, wait for one to finish
    while len(CURRENT_RUNNING) >= MAX_SUBPROCESS:
        for process in CURRENT_RUNNING:
            if process.poll() is not None:
                CURRENT_RUNNING.remove(process)
                break
    # Run the reasoner
    process = subprocess.Popen(
        COMMAND_PIPE + [file, result_dir / file.name]
    )
    CURRENT_RUNNING.append(process)

# Wait for all subprocesses to finish
for process in CURRENT_RUNNING:
    process.wait()