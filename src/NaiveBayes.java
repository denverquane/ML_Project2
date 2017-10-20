import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.*;

public class NaiveBayes
{
  //Total number of columns in the training file
  private static final int COLS = 61190;

  //Total number of unique words in the vocabulary
  private static final int UNIQUE_WORDS = COLS-2;

  //total number of training document records
  private static final int ROWS = 12000;

  //Should detailed information about the misclassifications be printed?
  private static final boolean PRINT_MISCLASSIFY = true;

  //This is the default value of Beta, which depends on the size of the alphabet
  private static double BETA = 1.0 + (1.0 / (double)(UNIQUE_WORDS));

  //Simple string array of all words found in the dictionary/vocabulary (only really used for question 6)
  private static final String[] VOCABULARY_ARRAY = getVocabArrFromFile("data/vocabulary.txt");


  /**
   * This determines if the program should be training and validating with the same dataset, or if we want to train with
   * all of the training data, and then test with testing.csv (which will be output to out.csv)
   */
  private static boolean CROSS_VALIDATE = false;

  /**
   * Main function for the Naive Bayes classifier
   * @param args Expects no arguments, or two arguments that specify the percentage of documents to use for training
   *             Ex: "-cross 50" will train with 50% of the training data, and test with the other 50%
   */
  public static void main(String[] args){
    NaiveBayes naiveBayes = new NaiveBayes();
    if(args.length > 1 && args[0].equals("-cross")){
      CROSS_VALIDATE = true;
      System.out.println("Cross-validating with " + Integer.parseInt(args[1]));
      naiveBayes.trainAndClassify(Integer.parseInt(args[1]));
    }
    else naiveBayes.trainAndClassify(100);

  }

  /**
   * Basic top-level function to open training file, organize data and compute NB estimates, and then classify data
   * according to the percentage of
   * @param percentToTrain integer value between 0 and 100 that should be used for determing how many documents to use
   *                       for training
   */
  private void trainAndClassify(int percentToTrain){
    File csv = new File("data/training.csv");

    //what ratio of the data we should be training with (ex: input 50 => percent = 0.5)
    double percent = (double)percentToTrain / 100.0;

    //total amount of documents we should use for training
    int maxDocuments = (int)(percent * 12000.0);

    //lines of the training data read in so far
    int totalDocs = 0;

    //how many inputs of those used for training are belonging to what class (of the 20). The class is the index
    int[] numInputsOfClass = new int[20];

    //Total record of how many times a specific word is found within a specific class (used for MLE)
    int[][] instancesOfWordsPerClass = new int[20][UNIQUE_WORDS];

    try
    {
      Scanner sc = new Scanner(csv);

      //keep reading input lines until the file is done, or we have enough to train with
      while(sc.hasNext() && totalDocs < maxDocuments){
        String s = sc.next();
        String[] ls = s.split(",");

        //get the class from the training data
        int clas = Integer.parseInt(ls[COLS-1]) - 1;

        for(int i = 1; i < COLS-1; i++){
          int elem =  Integer.parseInt(ls[i]);
          if(elem > 0) {
            instancesOfWordsPerClass[clas][i-1] += elem;
            //Add to the classes' record for this word, based on how many times it occurs
          }
        }
        totalDocs++; //finished with an input line
        numInputsOfClass[clas]++; //this input was part of the "clas" newsgroup, so add 1
      }

      System.out.println("Trained with a total of " + totalDocs + " documents");

      //get an array of the total number of words found in each class
      int[] totalWordsPerClass = getTotalWordsInClass(instancesOfWordsPerClass);

      //gets the log of the prior estimates (how often documents belong to each class based just on their count)
      double[] logPriors = getLogPriors(numInputsOfClass);


      double[][] likelihoods = getLikelihoods(totalWordsPerClass, instancesOfWordsPerClass);

      likelihoods = calculateEntropiesAndModifyLikelihoods(likelihoods, logPriors);

      classifyTestData(logPriors, likelihoods, maxDocuments);

    } catch (FileNotFoundException | UnsupportedEncodingException e)
    {
      e.printStackTrace();
    }
  }

  /**
   * Gets the total amount of times that words are found to belong to a newsgroup
   * @param wordInstancesPerClass (20x61188) array of word counts per class
   * @return a summed array of the total number of words are found in a newsgroup
   *      ex: if "athiesm" has 5 "no" and 2 "god", it would sum to 7 for the class "athiesm"
   */
  private int[] getTotalWordsInClass(int[][] wordInstancesPerClass){
    int[] returnArr = new int[20];

    for(int news = 0; news < 20; news++){
      for(int word = 0; word < UNIQUE_WORDS; word++){
        if(wordInstancesPerClass[news][word] > 1) returnArr[news] += wordInstancesPerClass[news][word];
      }
    }
    return returnArr;
  }

  /**
   * Simple function to take log base 2 of the input (used for entropy)
   * @param x input to take log2 of
   * @return log base 2 of input x
   */
  private static double log2(double x)
  {
    return Math.log(x) / Math.log(2);
  }

  /**
   * Looks at the unique words per newsgroup, and the TOTAL word occurrences per newsgroup, and calculates the
   * conditional probabilities of a given word belonging to a specific newsgroup (calculates P(news | word) for all
   * newsgroups and words)
   * @param wordsPerClass simple counter of how many words are found in each newsgroup
   * @param wordOccurrencesInClass how many occurrences of each individual word are found per newsgroup
   * @return (20x61188)ixj array of the likelihood that word j belongs to class/newsgroup i
   */
  private double[][] getLikelihoods(int[] wordsPerClass, int[][] wordOccurrencesInClass)
  {
    double[][] array = new double[20][UNIQUE_WORDS];

    for (int newsGroup = 0; newsGroup < 20; newsGroup++)
    {
      int totalClassWords = wordsPerClass[newsGroup];
      for (int word = 0; word < UNIQUE_WORDS; word++)
      {
        double likelihood = ((double) wordOccurrencesInClass[newsGroup][word] + (BETA - 1.0)) /
            ((double) totalClassWords + ((double) UNIQUE_WORDS * (BETA - 1.0)));
        //calculates P( X | Y )

        array[newsGroup][word] = likelihood;
      }
    }
    return array;
  }

  class IntDoublePair {
    final int i;
    final double likeli;

    IntDoublePair(int i, double likeli){this.i = i; this.likeli = likeli;}
    @Override
    public String toString(){ return "Word " + i + " has entropy " + likeli; }
  }

  /**
   * Used for printing the words found in the ranking method to answer questions 5-7
   */
  private static final boolean printRankingMethod = false;

  /**
   * This function calculates the entropies for all words across the 20 newsgroups. It then considers if the entropy
   * is too high to use for reliably classifying (a small filter to improve NB accuracy by ~0.3%), and then scales
   * likelihoods for these words proportional to their classes' probability (Priori estimate)
   * @param likeliHoods the likelihoods of classes based on words (P (Y | X))
   * @param logPriors the prior knowledge for all 20 groups, in log form (P (Y))
   * @return adjusted probabilities for entries with high entropy that should be adjusted based on priori knowledge
   */
  private double[][] calculateEntropiesAndModifyLikelihoods(double[][] likeliHoods, double[] logPriors){
    TreeSet<IntDoublePair> sortedTree = new TreeSet<>((o1, o2) -> Double.compare(o2.likeli, o1.likeli));

    for(int word = 0; word < UNIQUE_WORDS; word++){
      double wordEntropy = 0.0;

      //get the entropy of the conditional probability of a word, across all newsgroups/classes
      for(int group = 0; group < 20; group++){
        wordEntropy += -likeliHoods[group][word] * log2(likeliHoods[group][word]);
      }

      //Scale down the likelihood of choosing words that are found very frequently
      //this improves the accuracy by ~0.3% for Kaggle test data
      if(wordEntropy > 0.5){
        //System.out.println("HIGH ENTROPY FOR " + (word+1));
        for(int group = 0; group < 20; group++){
          //System.out.println("previous: " + likeliHoods[group][word] + " prior: " + Math.exp(logPriors[group]));
          likeliHoods[group][word] += Math.exp(logPriors[group]);
        }
      }
      if(wordEntropy > (1.0 / 20.0)) //only pick those that are above  ~1 instance per 20 documents
      sortedTree.add(new IntDoublePair(word, 5.0 - wordEntropy)); //flip the scale (chose the low entropy words)
    }

    //Get the top 100 ranking words from the sorted set, and print them
    if(printRankingMethod){
      int count = 0;
      for(IntDoublePair s : sortedTree){
        if(count == 100) break;
        else {
          count++;
          System.out.println(count + ": " + VOCABULARY_ARRAY[s.i]);
        }
      }
    }
    return likeliHoods; //returns the array with some modified entries, if their probabilities have changed as per
                        // the entropy scaling
  }

  /**
   * This function adds all the likelihoods in a row to obtain the MAP estimate for a document. This is the fundamental
   * operation for Naive Bayes; by adding all these probabilities, and our Prior knowledge estimate, we obtain a rough
   * idea of the likelihood of our input being classified in a certain newsgroup, based on the frequency of words in the
   * input itself
   * @param logPriors Log values of our prior knowledge, per document class (array is 20 long, 1 per newsgroup)
   * @param likelihoods (20x61188) array of the likelihoods of all words belonging to a specific class, as calculated in
   *                    "getLikelihoods"
   * @param inputWords Sequence of 61188 integers representing the frequency of the vocab. words appearing in the
   *                   document that we wish to classify
   * @return 20-long array representing the probabilities that the input belongs to a newsgroup. We should then pick the
   *          highest probability to determine which class the input likely belongs to
   */
  private double[] sumAllRowLikelihoodsApplyMAP(double[] logPriors, double[][] likelihoods, int[] inputWords){
    double[] returnArr = new double[20];
    for(int group = 0; group < 20; group++) {
      returnArr[group] += logPriors[group]; // P(newsgroup) (This is the MLE)
      for (int i = 0; i < UNIQUE_WORDS; i++) {
        if(inputWords[i] > 0) {
          returnArr[group] += (Math.log(likelihoods[group][i]) * inputWords[i]); //adds all P(x_i | newsgroup) (addition b/c Log)
        }
      }
    }
    return returnArr;
  }

  /**
   * Prior knowledge looks at all the data, and gets the probability of a document being in a class just on random
   * chance (in essence: # of documents in class X / # of documents total)
   * @param counts An array comprised of the total number of documents that are classified for a specific class
   *               (so if 10 documents are in the 1st class (Athiesm), then counts[0] would be 10
   * @return Calculated prior knowledge, which is a basic probabilstic estimate of how often documents are classified to
   *          a specific newsgroup
   */
  private double[] getLogPriors(int[] counts){
    double[] priors = new double[20];
    for(int i = 0; i < 20; i++){
      priors[i] = Math.log(counts[i]) - Math.log(ROWS);
    }
    return priors;
  }

  /**
   * Classify all input test data based on the prior knowledge, likelihood array, and what line the testing data starts
   * on (if the training data is being cross-validated)
   * @param logPriors log values of priori knowledge for all 20 classes
   * @param likelihoods 20x61188 array of P( Y | X ) estimates to be used for classification
   * @param startingLine the line of the input file that we should start at, for testing (after training lines previous)
   * @throws FileNotFoundException
   * @throws UnsupportedEncodingException
   */
  private void classifyTestData(double[] logPriors, double[][] likelihoods, int startingLine)
      throws FileNotFoundException, UnsupportedEncodingException
  {
    Scanner sc2;
    PrintWriter writer = null;
    int[][] confusionMatrix = null;
    int totalEvaluated = 0;
    int totalCorrect = 0;
    int count = 0;

    if(CROSS_VALIDATE) {
      sc2 = new Scanner(new File("data/training.csv"));
      confusionMatrix = new int[20][20];
    }
    else
    {
      sc2 = new Scanner(new File("data/testing.csv"));
      writer = new PrintWriter("out.csv", "UTF-8");
      writer.println("id,class");
    }


    while(sc2.hasNext()){
      String s2 = sc2.next();
      count++;
      if(count > startingLine || startingLine == 12000 || !CROSS_VALIDATE)
      {
        totalEvaluated++;
        String ls2[] = s2.split(",");
        int[] input = new int[UNIQUE_WORDS];
        int actualClass;

        for (int i = 1; i < COLS - 1; i++)
        {
          int elem = Integer.parseInt(ls2[i]);
          input[i - 1] = elem;
        }

        //now use the input data to estimate the class
        double[] classProbs = sumAllRowLikelihoodsApplyMAP(logPriors, likelihoods, input);

        int bestClass = getBestClass(classProbs);
        //System.out.println("ID: " + ls2[0] + "Class: " + clas);
        if (!CROSS_VALIDATE) writer.println(ls2[0] + "," + bestClass);
        else
        {
          actualClass = Integer.parseInt(ls2[COLS - 1]);
          if (actualClass > 0 && bestClass > 0) confusionMatrix[bestClass - 1][actualClass - 1]++;
          if (actualClass == bestClass)
          {
            totalCorrect++;
          } else
          {
            //          if(PRINT_MISCLASSIFY) System.out.println("ID: " + ls2[0] + " mismatch -> Predicted " +
//              bestClass + " vs Actual " + actualClass);
          }
        }
      }
    }
    if(!CROSS_VALIDATE) {
      writer.close();
      System.out.println("Completed with Beta = " + BETA);
    }
    else{
      if(PRINT_MISCLASSIFY)
      {
        for (int i = 0; i < 20; i++)
        {
          if (i < 9) System.out.print("Predicted " + (i + 1) + " : ");
          else System.out.print("Predicted " + (i + 1) + ": ");
          for (int j = 0; j < 20; j++)
          {
            int val = confusionMatrix[i][j];
            if (val < 10) System.out.print(val + "   ");
            else if(val < 100) System.out.print(val + "  ");
            else System.out.print(val + " ");
          }
          System.out.println();
        }
      }
      System.out.println("\nPredicted " + totalCorrect + "/" + totalEvaluated + " Correctly (" +
          100.0*((double)totalCorrect/(double)totalEvaluated) + "%) with Beta = " + (BETA - 1.0));
    }
  }

  /**
   * Open the vocabulary file, and load the  words found into a string array
   * @param filePath relative path to the vocabulary text file
   * @return array of all strings found in the vocabulary
   */
  private static String[] getVocabArrFromFile(String filePath)
  {
    String[] vocab = new String[UNIQUE_WORDS];

    Scanner sc = null;
    try
    {
      sc = new Scanner(new File(filePath));
    } catch (FileNotFoundException e)
    {
      e.printStackTrace();
      return vocab;
    }
    int count = 0;

    while(sc.hasNext()){
      String word = sc.next();

      vocab[count] = word;
      count++;
    }
    return vocab;
  }

  /**
   * Just look through the class probabilities and pick the maximum
   * @param classProbs probability of classifying as a certain class
   * @return index of the highest probability class
   */
  private int getBestClass(double[] classProbs){
    double max = Double.NEGATIVE_INFINITY;
    int clas = -1;
    for(int i = 0; i < 20; i++){
      //System.out.println("Prob: " + classProbs[i]);
      if(classProbs[i] > max){
        max = classProbs[i];
        clas = i + 1;
      }
    }
    return clas;
  }
}
