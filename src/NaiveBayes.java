import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class NaiveBayes
{
  public static final int COLS = 61190;
  public static final int UNIQUE_WORDS = COLS-2;
  public static final int ROWS = 12000;

  public static void main(String[] args){
    NaiveBayes csv = new NaiveBayes();
    csv.parse();
  }

  private void parse(){
    File csv = new File("data/training.csv");
    ArrayList<TrainingInput>[] allInputsOfClass = new ArrayList[20];
    int[][] instancesOfWordsPerClass = new int[20][UNIQUE_WORDS];

    for(int i =0; i < 20; i++){
      allInputsOfClass[i] = new ArrayList<>();
    }

    try
    {
      Scanner sc = new Scanner(csv);

      while(sc.hasNext()){
        String s = sc.next();
        String[] ls = s.split(",");
        int[] counts = new int[UNIQUE_WORDS];
        int sum = 0;
        int clas = Integer.parseInt(ls[COLS-1]) - 1;

        for(int i = 1; i < COLS-1; i++){
          int elem =  Integer.parseInt(ls[i]);
          if(elem > 0) instancesOfWordsPerClass[clas][i-1]++;
          counts[i-1] = elem;
          sum += elem;
        }

        TrainingInput input = new TrainingInput(Integer.parseInt(ls[0]), counts, sum, clas);
        allInputsOfClass[clas].add(input);
        System.out.println(input);
      }
      int[] uniqueWordsPerClass = getUniqueWordsInClass(instancesOfWordsPerClass);

      double[] prior = getPriors(allInputsOfClass); //gets the prior estimates

      double[][] likelihoods = getLikelihoods(uniqueWordsPerClass, instancesOfWordsPerClass);
        //likelihood array of a word being in a particular class



      //now use the input data to estimate the class
//      double[] classLogSums = getLikelihoodRowLogSums();
//
//      double[] classProbs = getProbsOfClasses(prior, classLogSums);

    } catch (FileNotFoundException e)
    {
      e.printStackTrace();
    }
  }

  private int[] getUniqueWordsInClass(int[][] classContainsWord){
    int[] returnArr = new int[20];

    for(int news = 0; news < 20; news++){
      for(int word = 0; word < UNIQUE_WORDS; word++){
        if(classContainsWord[news][word] > 0) returnArr[news]++;
      }
    }
    return returnArr;
  }

  private double[][] getLikelihoods(int[] uniqueWordsPerClass, int[][] wordOccurrencesInClass){
    double[][] array = new double[20][UNIQUE_WORDS];

    for(int newsGroup = 0; newsGroup < 20; newsGroup++){
      int uniqueWords = uniqueWordsPerClass[newsGroup];
      for(int word = 0; word < UNIQUE_WORDS; word++){
        double likelihood = (double)wordOccurrencesInClass[newsGroup][word] / (double)(uniqueWords + UNIQUE_WORDS);
        array[newsGroup][word] = Math.log(likelihood);
      }
    }
    return array;
  }

  private double[] getProbsOfClasses(double[] priors, double[] logLikely){
    for(int i = 0; i < 20; i++){
      logLikely[i] += Math.log(priors[i]);
    }
    return logLikely;
  }

  private double[] getLikelihoodRowLogSums(double[][] likelihoods){
    double[] returnArr = new double[20];
    for(int news = 0; news < 20; news++)
    {
      for (int i = 0; i < UNIQUE_WORDS; i++)
      {
        returnArr[news] += likelihoods[news][i];
      }
    }
    return returnArr;
  }

  private double[] getPriors(ArrayList[] counts){
    double[] priors = new double[20];
    for(int i = 0; i < 20; i++){
      priors[i] = (double)counts[i].size() / (double)ROWS;
      System.out.println("I: " + i + ": " + priors[i]);
    }
    return priors;
  }

  class TrainingInput {
    private int id;
    private int[] counts;
    private int sum;
    private int clas;

    TrainingInput(int id, int[] counts, int sum, int clas)
    {
      this.id = id;
      this.counts = counts;
      this.sum = sum;
      this.clas = clas;
    }
    int getSum(){
      return sum;
    }

    @Override
    public String toString(){
      return "Id : " + id + " has " + sum + " words in class " + clas;
    }
  }
}
