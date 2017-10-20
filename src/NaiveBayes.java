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


  private static final int ROWS = 12000;
  private static final boolean PRINT_MISCLASSIFY = true;

  //This is the default value of Beta, which depends on the size of the alphabet
  private static double BETA = 1.0 + (1.0 / (double)(UNIQUE_WORDS));

  private static final String[] VOCABULARY_ARRAY = getVocabArrFromFile("data/vocabulary.txt");

  //private static int[][] globalFreq = new int[20][UNIQUE_WORDS];


  /**
   * This determines if the program should be training and validating with the same dataset, or if we want to train with
   * all of the training data, and then test with testing.csv (which will be output to out.csv)
   */
  private static boolean CROSS_VALIDATE = false;

  public static void main(String[] args){
    NaiveBayes naiveBayes = new NaiveBayes();
    if(args.length > 1 && args[0].equals("-cross")){
      CROSS_VALIDATE = true;
      System.out.println("Cross-validating with " + Integer.parseInt(args[1]));
      naiveBayes.trainAndClassify(Integer.parseInt(args[1]));
    }
    else naiveBayes.trainAndClassify(100);

  }

  private void trainAndClassify(int percentToTrain){
    File csv = new File("data/training.csv");

    //what ratio of the data we should be training with (ex: input 50 => percent = 0.5)
    double percent = (double)percentToTrain / 100.0;

    //total amount of documents we should use for training
    int maxDocuments = (int)(percent * 12000.0);


    int totalDocs = 0;
    int[] numInputsOfClass = new int[20];
    int[][] instancesOfWordsPerClass = new int[20][UNIQUE_WORDS];

    try
    {
      Scanner sc = new Scanner(csv);

      while(sc.hasNext() && totalDocs < maxDocuments){
        String s = sc.next();
        String[] ls = s.split(",");

        int clas = Integer.parseInt(ls[COLS-1]) - 1;

        for(int i = 1; i < COLS-1; i++){
          int elem =  Integer.parseInt(ls[i]);
          if(elem > 0) {
            instancesOfWordsPerClass[clas][i-1] += elem;
          }
        }
        totalDocs++;
        numInputsOfClass[clas]++;
      }


      System.out.println("Trained with a total of " + totalDocs + " documents");

      int[] uniqueWordsPerClass = getTotalWordsInClass(instancesOfWordsPerClass);

      double[] logPriors = getLogPriors(numInputsOfClass);
      //gets the prior estimates

      double[][] likelihoods = getLikelihoods(uniqueWordsPerClass, instancesOfWordsPerClass);

      likelihoods = getEntropiesOfLikelihoods(likelihoods, logPriors, instancesOfWordsPerClass);

      classifyTestData(logPriors, likelihoods, maxDocuments);

    } catch (FileNotFoundException | UnsupportedEncodingException e)
    {
      e.printStackTrace();
    }
  }

  /**
   * Gets the total amount of times that words are found to belong to a newsgroup
   * @param wordInstancesPerClass 
   * @return
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

  private static double log2(double x)
  {
    return Math.log(x) / Math.log(2);
  }

  /**
   * Looks at the unique words per newsgroup, and the TOTAL word occurrences per newsgroup, and calculates the
   * conditional probabilities of a given word belonging to a specific newsgroup (calculates P(news | word) for all
   * newsgroups and words)
   * @param uniqueWordsPerClass simple counter of how many unique words are found in each newsgroup
   * @param wordOccurrencesInClass how many occurrences of every word are found per newsgroup
   * @return
   */
  private double[][] getLikelihoods(int[] uniqueWordsPerClass, int[][] wordOccurrencesInClass)
  {
    double[][] array = new double[20][UNIQUE_WORDS];

    for (int newsGroup = 0; newsGroup < 20; newsGroup++)
    {
      int uniqueWords = uniqueWordsPerClass[newsGroup];
      for (int word = 0; word < UNIQUE_WORDS; word++)
      {
        double likelihood = ((double) wordOccurrencesInClass[newsGroup][word] + (BETA - 1.0)) /
            ((double) uniqueWords + ((double) UNIQUE_WORDS * (BETA - 1.0)));
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

  private static final boolean printWeighted = true;

  private double[][] getEntropiesOfLikelihoods(double[][] likeliHoods, double[] logPriors, int[][] allOccurrences){
    TreeSet<IntDoublePair> sortedTree = new TreeSet<>((o1, o2) -> Double.compare(o2.likeli, o1.likeli));

    for(int word = 0; word < UNIQUE_WORDS; word++){
      double wordEntropy = 0.0;

      for(int group = 0; group < 20; group++){
        wordEntropy += -likeliHoods[group][word] * log2(likeliHoods[group][word]);
      }


      if(wordEntropy > 0.5){
        System.out.println("HIGH ENTROPY FOR " + (word+1));
        for(int group = 0; group < 20; group++){
          //System.out.println("previous: " + likeliHoods[group][word] + " prior: " + Math.exp(logPriors[group]));
          likeliHoods[group][word] += Math.exp(logPriors[group]);
        }
      }
      if(wordEntropy > (1.0 / 20.0)) //only pick those that are above 1 instance per 20 documents
      sortedTree.add(new IntDoublePair(word, 5.0 - wordEntropy)); //flip the scale (chose the low entropy words)
    }
    if(printWeighted){
      int count = 0;
      for(IntDoublePair s : sortedTree){
        if(count == 100) break;
        else {
          count++;
          int sum = 0;
          for(int n = 0; n < 20; n++){
            sum += allOccurrences[n][s.i];
          }
          System.out.println(count + ": " + VOCABULARY_ARRAY[s.i]);

        }
      }
    }
    return likeliHoods;
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
   * @param logPriors
   * @param likelihoods
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
   * @param classProbs
   * @return
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
