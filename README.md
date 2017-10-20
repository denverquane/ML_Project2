# Naive Bayes
Denver Quane and Matthew McChesney CS429/CS529 Naive Bayes

To run the program simply run the included "runNaiveBayes.sh"
script inside of a Bash or cmd shell (assuming all files have been 
extracted from the .zip archive),

--OR--

compile the .java source file found in "src" using "javac" (or build tools
included in any Java IDE such as IntelliJ or Eclipse), and then run the 
program by using the main class found in NaiveBayes.java.

Example: Navigate to /src "javac *.java"  to compile all files, followed by 
"java NaiveBayes" + optional command line argument explained below.
 
This program comes with an option of a command line argument to enable cross 
validation "-cross x", where x is a number specified by the user that 
represents the percent of the training data to be cross-validated. 
 
Example: "java NaiveBayes -cross 50". This would run the algorithm training 
on 50% of training.csv, and testing on 50% of the file.

By default (no command args) the program trains on the whole training set,
and tests on the testing.csv file. The result is output to out.csv.
 
For any further questions or concerns, contact Denver Quane at dquane@unm.edu,
or Matthew Mcchesney at mmcchesney@unm.edu.

