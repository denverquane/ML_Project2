import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.*;

public class NaiveBayes
{
  private static final int COLS = 61190;
  private static final int UNIQUE_WORDS = COLS-2;
  private static final int ROWS = 12000;
  private static final boolean PRINT_MISCLASSIFY = true;
  //private static double BETA = 0.0000001;
  private static double BETA = 1.0 + (1.0 / (double)(UNIQUE_WORDS));

  private static final boolean CROSS_VALIDATE = true;

  public static void main(String[] args){
    NaiveBayes csv = new NaiveBayes();

    if(CROSS_VALIDATE){
      csv.trainAndClassify(75);
//      csv.trainAndClassify(25);
//      csv.trainAndClassify(50);
//      csv.trainAndClassify(75);
//      csv.trainAndClassify(100);
    }
    else csv.trainAndClassify(100);
  }

  private void trainAndClassify(int percentToTrain){
    File csv = new File("data/training.csv");
    double percent = (double)percentToTrain / 100.0;
    int maxDocuments = (int)(percent * 12000.0);
    int totalDocs = 0;
    int[] numInputsOfClass = new int[20];
    int[][] instancesOfWordsPerClass = new int[20][UNIQUE_WORDS];
    //int[] totalNumberOfWordInstance = new int[UNIQUE_WORDS];
    //int[] uniqueDocsWordIsFoundIn = new int[UNIQUE_WORDS];
    /**
     * THESE INDICES ARE BACKWARDS, DO NOT CONFUSE THIS! Backwards indices should make traversing the complete records
     * for one single word across all inputs faster (b/c of row incrementing as opposed to column incrementing)
     */
    //int[][] completeWordFreqRecord = new int[UNIQUE_WORDS][12000];

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
            instancesOfWordsPerClass[clas][i-1]++;
            //uniqueDocsWordIsFoundIn[i-1]++;
          }

          //completeWordFreqRecord[i-1][totalDocs] = elem;
          //totalNumberOfWordInstance[i-1] += elem;
        }
        totalDocs++;
        numInputsOfClass[clas]++;
      }


      System.out.println("Trained with a total of " + totalDocs + " documents");

      int[] uniqueWordsPerClass = getUniqueWordsInClass(instancesOfWordsPerClass);

      double[] logPriors = getLogPriors(numInputsOfClass);
      //gets the prior estimates

      double[][] likelihoods = getLikelihoods(uniqueWordsPerClass, instancesOfWordsPerClass);

      likelihoods = getEntropiesOfLikelihoods(likelihoods, logPriors);

      classifyTestData(logPriors, likelihoods, maxDocuments);

    } catch (FileNotFoundException | UnsupportedEncodingException e)
    {
      e.printStackTrace();
    }
  }

  /**
   * Gets the number of unique words found in each newsgroup
   * @param wordInstancesPerClass 
   * @return
   */
  private int[] getUniqueWordsInClass(int[][] wordInstancesPerClass){
    int[] returnArr = new int[20];

    for(int news = 0; news < 20; news++){
      for(int word = 0; word < UNIQUE_WORDS; word++){
        if(wordInstancesPerClass[news][word] > 0) returnArr[news]++;
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

  private static final boolean printWeighted = false;

  private double[][] getEntropiesOfLikelihoods(double[][] likeliHoods, double[] logPriors){
    TreeSet<IntDoublePair> sortedTree = new TreeSet<>((o1, o2) -> Double.compare(o2.likeli, o1.likeli));

    double[][] entropies = new double[20][UNIQUE_WORDS];


    for(int word = 0; word < UNIQUE_WORDS; word++){
      double wordEntropy = 0.0;

      for(int group = 0; group < 20; group++){
        entropies[group][word] = -likeliHoods[group][word] * log2(likeliHoods[group][word]);

        wordEntropy += entropies[group][word];
      }

      if(wordEntropy > 0.5){
        System.out.println("HIGH ENTROPY FOR " + (word+1));
        for(int group = 0; group < 20; group++){
          System.out.println("previous: " + likeliHoods[group][word] + " prior: " + Math.exp(logPriors[group]));

          likeliHoods[group][word] += Math.exp(logPriors[group]);
        }
      }

      sortedTree.add(new IntDoublePair(word + 1, 5.0 - wordEntropy));
    }
    if(printWeighted){
      int count = 100;
      for(IntDoublePair s : sortedTree){
        if(count == 0) break;
        else {
          count--;
          System.out.println(s);
        }
      }
    }
    return likeliHoods;
  }

  private double[] getLikelihoodRowLogSums(double[] logPriors, double[][] likelihoods, int[] inputWords){
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
   * Prior knowledge looks at all the data, and gets the probability of a document being in a class just on random chance
   * (in essence: # of documents in class X / # of documents total)
   * @param counts
   * @return
   */
  private double[] getLogPriors(int[] counts){
    double[] priors = new double[20];
    for(int i = 0; i < 20; i++){
      priors[i] = Math.log(counts[i]) - Math.log(ROWS);
      //System.out.println("I: " + i + ": " + priors[i]);
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
    PrintWriter writer;
    int[][] confusionMatrix;
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
        double[] classProbs = getLikelihoodRowLogSums(logPriors, likelihoods, input);

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
