import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Scanner;

public class NaiveBayes
{
  public static final int COLS = 61190;
  public static final int UNIQUE_WORDS = COLS-2;
  public static final int ROWS = 12000;

  public static void main(String[] args){
    NaiveBayes csv = new NaiveBayes();
    csv.trainAndClassify();
  }

  private void trainAndClassify(){
    File csv = new File("data/training.csv");
    int[] numInputsOfClass = new int[20];
    int[][] instancesOfWordsPerClass = new int[20][UNIQUE_WORDS];

    try
    {
      Scanner sc = new Scanner(csv);

      while(sc.hasNext()){
        String s = sc.next();
        String[] ls = s.split(",");

        int clas = Integer.parseInt(ls[COLS-1]) - 1;

        for(int i = 1; i < COLS-1; i++){
          int elem =  Integer.parseInt(ls[i]);
          if(elem > 0) instancesOfWordsPerClass[clas][i-1]++;
        }
        numInputsOfClass[clas]++;
      }

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
        double likelihood = (double)(wordOccurrencesInClass[newsGroup][word] + 1) / (double)(uniqueWords + UNIQUE_WORDS);
        array[newsGroup][word] = Math.log(likelihood);
      }
    }
    return array;
  }

  private double[] getLikelihoodRowLogSums(double[] logPriors, double[][] likelihoods, int[] inputWords){
    double[] returnArr = new double[20];
    for(int news = 0; news < 20; news++) {
      returnArr[news] += logPriors[news]; // P(newsgroup)
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


  private void classifyTestData(double[] logPriors, double[][] likelihoods) throws FileNotFoundException, UnsupportedEncodingException
  {
    Scanner sc2 = new Scanner(new File("data/testing.csv"));
    PrintWriter writer = new PrintWriter("out.csv", "UTF-8");
    writer.println("id,class");

    while(sc2.hasNext()){
      String s2 = sc2.next();
      String ls2[] = s2.split(",");
      int[] input = new int[UNIQUE_WORDS];

      for(int i = 1; i < COLS-1; i++){
        int elem =  Integer.parseInt(ls2[i]);
        input[i-1] = elem;
      }

      //now use the input data to estimate the class
      double[] classProbs = getLikelihoodRowLogSums(logPriors, likelihoods, input);

      int bestClass = getBestClass(classProbs);
      //System.out.println("ID: " + ls2[0] + "Class: " + clas);
      writer.println(ls2[0] + "," + bestClass);
    }
    writer.close();
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
