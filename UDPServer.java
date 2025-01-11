import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

public class UDPServer {
    public static void main(String[] args) throws IOException {
        
        DatagramSocket socket = new DatagramSocket(9000); 
        byte[] inServer = new byte[1024];

        System.out.println("Server is running...");

        while (true) {
           
            DatagramPacket rcvPkt = new DatagramPacket(inServer, inServer.length);
            
            socket.receive(rcvPkt);
  
            int length = rcvPkt.getLength();
            byte[] receivedData = Arrays.copyOf(rcvPkt.getData(), length);

            System.out.println("Packet Received! Bytes: " + length);

            float[] floatArray = byteArrayToFloatArray(receivedData);
  
            int numTags = floatArray.length / 8;
            System.out.println("Detected " + numTags + " tags:");
            
            for (int i = 0; i < numTags; i++) {
                System.out.print("Tag " + (i + 1) + " [X, Y, Z, Qw, Qx, Qy, Qz, ID]: ");
                for (int j = 0; j < 8; j++) {
                    System.out.print(floatArray[i * 8 + j] + " ");
                }
                System.out.println();
            }
        }
    }

    private static float[] byteArrayToFloatArray(byte[] bytes) {
        int numFloats = bytes.length / Float.BYTES;
        float[] floatArray = new float[numFloats];

        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN); // Ensure correct byte order

        for (int i = 0; i < numFloats; i++) {
            floatArray[i] = buffer.getFloat();
        }

        return floatArray;
    }
}
