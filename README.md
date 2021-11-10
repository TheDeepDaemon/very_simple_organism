# very_simple_organism
My capstone project, I am trying to emulate how a organism with a simple CNS might learn and navigate. 
<br /><br />
How to get it running:
<br />

Python:
* Create a Python virtual environment nearby (or somewhere) that you can select as an interpreter. 
* Use "pip install -r requirements.txt" to install the dependencies to your venv; the requirements.txt file is included in the repository. 
* Run the project from main.py. 
<br /><br />

C++:
* The C++ code is accessed by Python from the file "cppfunctions.py"
	* This file acts as an interface between the two parts of the project.
* There should already be a "cpp_code\main.so" file.
* If you modify the C++ code, use "g++ -O2 -fPIC -shared -o ..\..\bin\main.so main.cpp" from command prompt (from the "cpp_code" folder) to compile it again.
	* I used g++ 64 bit.
	* You can use it with another optimization setting if needed.
	* Make sure there is a "bin" folder for it to output to.

