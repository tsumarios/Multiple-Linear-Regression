/******************************************************************************************************************
 *  
 *                                      MULTIPLE LINEAR REGRESSION EXAMPLE
 *                                               by Mario Raciti
 *
 *  The following code, from https://introcs.cs.princeton.edu/java/97data/MultipleLinearRegression.java.html,
 *  is an adaptation to a specific multiple linear regression example with some modifications, so I don't own
 *  its rights. This is aimed to a statistic project relation at Univesity of Catania (Italy).
 *  
 *
 *  Compilation:  javac -classpath jama.jar:. MultipleLinearRegression.java
 *  Execution:    java  -classpath jama.jar:. MultipleLinearRegression
 *  Dependencies: jama.jar
 *  
 *  Compute least squares solution to X beta = y using Jama library.
 *  Assumes X has full column rank.
 *  
 *       http://math.nist.gov/javanumerics/jama/
 *       http://math.nist.gov/javanumerics/jama/Jama-1.0.1.jar
 *
 ******************************************************************************************************************/
 
import Jama.Matrix;
import Jama.QRDecomposition;
 
public class MultipleLinearRegression {
    private final int N;        // number of 
    private final int p;        // number of dependent variables
    private final Matrix beta;  // regression coefficients
    private double SSE;         // sum of squared
    private double SST;         // sum of squared
 
    private static double _leftInterval(double beta, double t, double S){
        return beta - (t*S);
    }
 
    private static double _rightInterval(double beta, double t, double S) {
        return beta + (t * S);
    }
 
    public MultipleLinearRegression(double[][] x, double[] y) {
        if (x.length != y.length) throw new RuntimeException("dimensions don't agree");
        N = y.length;
        p = x[0].length;
 
        Matrix X = new Matrix(x);
 
        // create matrix from vector
        Matrix Y = new Matrix(y, N);
 
        // find least squares solution
        QRDecomposition qr = new QRDecomposition(X);
        beta = qr.solve(Y);
 
        // mean of y[] values
        double sum = 0.0;
        for (int i = 0; i < N; i++)
            sum += y[i];
        double mean = sum / N;
 
        // total variation to be accounted for
        for (int i = 0; i < N; i++) {
            double dev = y[i] - mean;
            SST += dev*dev;
        }
 
        // variation not accounted for
        Matrix residuals = X.times(beta).minus(Y);
        SSE = residuals.norm2() * residuals.norm2();
 
    }
 
    public double beta(int j) {
        return beta.get(j, 0);
    }
 
    public double R2() {
        return 1.0 - SSE/SST;
    }
 
    public double getSSE(){
        return SSE;
    }
    public double getSST(){
        return SST;
    }
 
    // get confidence interval
    public static double[] confidenceInterval(double beta, double t, double S){
        double[] result = new double[2];
        double left = _leftInterval(beta, t, S), right = _rightInterval(beta, t, S);
        result[0] = (left < right) ? left : right;
        result[1] = (left > right) ? left : right;
        return result;
    }
 
    // Main
    public static void main(String[] args) {
 
        //                  x1    x2     x3   x4    x5
        double[][] x = { {  1,  3064,  1201,  10,  361 },
                         {  1,  3000,  1053,  11,  338 },
                         {  1,  3155,  1133,  19,  393 },
                         {  1,  3080,   970,   4,  467 },
                         {  1,  3245,  1258,  36,  294 },
                         {  1,  3267,  1386,  35,  225 },
                         {  1,  3080,   966,  13,  417 },
                         {  1,  2974,  1185,  12,  488 },
                         {  1,  3038,  1103,  14,  677 },
                         {  1,  3318,  1310,  29,  427 },
                         {  1,  3317,  1362,  25,  326 },
                         {  1,  3182,  1171,  28,  326 },
                         {  1,  2998,  1102,   9,  349 },
                         {  1,  3221,  1424,  21,  382 },
                         {  1,  3019,  1239,  16,  275 },
                         {  1,  3022,  1285,   9,  303 },
                         {  1,  3094,  1329,  11,  339 },
                         {  1,  3009,  1210,  15,  536 },
                         {  1,  3227,  1331,  21,  414 },
                         {  1,  3308,  1368,  24,  282 },
                         {  1,  3212,  1289,  17,  302 },
                         {  1,  3381,  1444,  25,  253 },
                         {  1,  3061,  1175,  12,  261 },
                         {  1,  3478,  1317,  42,  259 },
                         {  1,  3126,  1248,  11,  315 },
                         {  1,  3468,  1508,  43,  286 },
                         {  1,  3252,  1361,  26,  346 },
                         {  1,  3052,  1186,  14,  443 },
                         {  1,  3270,  1399,  24,  306 },
                         {  1,  3198,  1299,  20,  367 },
                         {  1,  2904,  1164,   6,  311 },
                         {  1,  3247,  1277,  19,  375 } };
 
        double[] y = { 1, -1, 1, -2, 2, 3, -2, -2, -3, 1, 2, -1, -2, 2, 0, 0, 0, -2, 1, 2, 1, 3, 0, 3, 1, 3, 1, -2, 2, 2, -2, 2 };
        MultipleLinearRegression regression = new MultipleLinearRegression(x, y);
        double[] confidenceBeta0 = confidenceInterval(regression.beta(0), -4.97, 5.384);
        double[] confidenceBeta1 = confidenceInterval(regression.beta(1), 3.97, 0.00193);
        double[] confidenceBeta2 = confidenceInterval(regression.beta(2), 3.06, 0.00144);
        double[] confidenceBeta3 = confidenceInterval(regression.beta(3), -0.92, 0.02574);
        double[] confidenceBeta4 = confidenceInterval(regression.beta(4), -3.88, 0.00154);
 
        /* Print results */
        System.out.println("\n>>Regression equation is:");
        System.out.println("y = "+ regression.beta(0) + " + "
                            + regression.beta(1) + " x2 + " + regression.beta(2) + " x3 + "
                            + regression.beta(3) + " x4 + " + regression.beta(4) + " x5\n");
        System.out.println(">>R^2 = " + (regression.R2())*100 + " %\n");
        System.out.println(">>SSE = " + regression.getSSE());
        System.out.println(">>SST = " + regression.getSST());
        System.out.println("\n>>Confidence intervals:\nbeta1: ["
                            + confidenceBeta0[0] + " , " + confidenceBeta0[1] + "]\nbeta2: ["
                            + confidenceBeta1[0] + " , " + confidenceBeta1[1] + "]\nbeta3: ["
                            + confidenceBeta2[0] + " , " + confidenceBeta2[1] + "]\nbeta4: ["
                            + confidenceBeta3[0] + " , " + confidenceBeta3[1] + "]\nbeta5: ["
                            + confidenceBeta4[0] + " , " + confidenceBeta4[1] + "]\n");
    }
 
}
