import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Scanner;

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
      for(int trainPercent = 10; trainPercent < 101; trainPercent += 10){
        csv.trainAndClassify(trainPercent);
      }
    }
    //if(CROSS_VALIDATE) for(BETA = 0.00001; BETA < 1.1; BETA *= 10.0)
    //csv.trainAndClassify(100);
    //else csv.trainAndClassify();
  }

  private void trainAndClassify(int percentToTrain){
    File csv = new File("data/training.csv");
    double percent = (double)percentToTrain / 100.0;
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
        totalDocs++;

        int clas = Integer.parseInt(ls[COLS-1]) - 1;

        for(int i = 1; i < COLS-1; i++){
          int elem =  Integer.parseInt(ls[i]);
          if(elem > 0) instancesOfWordsPerClass[clas][i-1]++;
        }
        numInputsOfClass[clas]++;
      }
      System.out.println("Trained with a total of " + totalDocs + " documents");

      int[] uniqueWordsPerClass = getUniqueWordsInClass(instancesOfWordsPerClass);

      double[] logPriors = getPriors(numInputsOfClass);
      //gets the prior estimates

      double[][] likelihoods = getLikelihoods(uniqueWordsPerClass, instancesOfWordsPerClass);
      //likelihood array of words being in a particular class


      classifyTestData(logPriors, likelihoods);

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

  /**
   * Looks at the unique words per newsgroup, and the TOTAL word occurrences per newsgroup, and calculates the
   * conditional probabilities of a given word belonging to a specific newsgroup (calculates P(news | word) for all
   * newsgroups and words)
   * @param uniqueWordsPerClass simple counter of how many unique words are found in each newsgroup
   * @param wordOccurrencesInClass how many occurrences of every word are found per newsgroup
   * @return
   */
  private double[][] getLikelihoods(int[] uniqueWordsPerClass, int[][] wordOccurrencesInClass){
    double[][] array = new double[20][UNIQUE_WORDS];

    for(int newsGroup = 0; newsGroup < 20; newsGroup++){
      int uniqueWords = uniqueWordsPerClass[newsGroup];
      for(int word = 0; word < UNIQUE_WORDS; word++){
        double likelihood = ((double)wordOccurrencesInClass[newsGroup][word] + (BETA - 1.0)) /
                            ((double)uniqueWords + ((double)UNIQUE_WORDS * (BETA - 1.0)));
        array[newsGroup][word] = Math.log(likelihood);
      }
    }
    return array;
  }

  private double[] getLikelihoodRowLogSums(double[] logPriors, double[][] likelihoods, int[] inputWords){
    double[] returnArr = new double[20];
    for(int news = 0; news < 20; news++) {
      returnArr[news] += logPriors[news]; // P(newsgroup) (This is the MLE)
      for (int i = 0; i < UNIQUE_WORDS; i++) {
        if(inputWords[i] > 0) {
          returnArr[news] += (likelihoods[news][i] * inputWords[i]); //adds all P(x_i | newsgroup) (addition b/c Log)
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
  private double[] getPriors(int[] counts){
    double[] priors = new double[20];
    for(int i = 0; i < 20; i++){
      priors[i] = Math.log(counts[i]) - Math.log(ROWS);
      //System.out.println("I: " + i + ": " + priors[i]);
    }
    return priors;
  }


  private void classifyTestData(double[] logPriors, double[][] likelihoods)
      throws FileNotFoundException, UnsupportedEncodingException
  {
    Scanner sc2;
    PrintWriter writer;
    int[][] confusionMatrix;
    int totalEvaluated = 0;
    int totalCorrect = 0;

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
      totalEvaluated++;
      String ls2[] = s2.split(",");
      int[] input = new int[UNIQUE_WORDS];
      int actualClass;

      for(int i = 1; i < COLS-1; i++){
        int elem =  Integer.parseInt(ls2[i]);
        input[i-1] = elem;
      }

      //now use the input data to estimate the class
      double[] classProbs = getLikelihoodRowLogSums(logPriors, likelihoods, input);

      int bestClass = getBestClass(classProbs);
      //System.out.println("ID: " + ls2[0] + "Class: " + clas);
      if(!CROSS_VALIDATE)  writer.println(ls2[0] + "," + bestClass);
      else{
        actualClass = Integer.parseInt(ls2[COLS-1]);
        if(actualClass > 0 && bestClass > 0) confusionMatrix[bestClass - 1][actualClass - 1]++;
        if(actualClass == bestClass){
          totalCorrect++;
        } else {
          //          if(PRINT_MISCLASSIFY) System.out.println("ID: " + ls2[0] + " mismatch -> Predicted " +
//              bestClass + " vs Actual " + actualClass);
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
          100.0*((double)totalCorrect/(double)totalEvaluated) + "%) with Beta = " + BETA);
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
