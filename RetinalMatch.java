import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;

public class RetinalMatch {
    public static void main(String[] args){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat src=Imgcodecs.imread("RIDB/RIDB/IM000001_1.jpg");
        Mat dst=new Mat();
        Mat gray = new Mat();
        Mat edges = new Mat();

        Imgproc.cvtColor(src, src, Imgproc.COLOR_RGB2GRAY);
        //Imgproc.cvtColor(dst,dst,Imgproc.COLOR_RGB2GRAY);

        //Increasing the contrast of the image
        src.convertTo(src, -1, 1.2, 50);

        //Adding smoothing
        Imgproc.medianBlur(src, edges, 19);
        Imgproc.GaussianBlur(edges, edges, new Size(11, 11), 0);

        //Imgproc.GaussianBlur(gray, edges, new Size(11, 11), 0);
        
        //Detecting the edges
        Imgproc.Canny(edges, edges, 4,12);

        //Copying the detected edges to the destination matrix
        src.copyTo(dst, edges);    

        //Setting to greyscale.
        Imgproc.threshold(dst, dst, 0, 1000, Imgproc.THRESH_BINARY);
        
        Imgcodecs.imwrite("IM000001_1_contrast_2.jpg",dst);

        //Some code from this source
        //https://www.tutorialspoint.com/java-example-demonstrating-canny-edge-detection-in-opencv
    }
}
