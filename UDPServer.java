import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class UDPServer {
    public static void main(String[] args) throws IOException {
        // Create the DatagramSocket bound to port 9000
        DatagramSocket socket = new DatagramSocket(9000);

        // Buffer to receive incoming packets
        byte[] inServer = new byte[28];

        System.out.println("Server is running...");

        while (true) {
            // Create a DatagramPacket to receive data
            DatagramPacket rcvPkt = new DatagramPacket(inServer, inServer.length);

            // Receive a packet (blocking call)
            socket.receive(rcvPkt);

            // Display receive
            System.out.println("Packet Received!");

            // Extract data from the received packet
            byte[] data = rcvPkt.getData();
            int length = rcvPkt.getLength();
            int numFloats = length / Float.BYTES;

            // Convert byte array to float array
            float[] floatArray = new float[numFloats];
            for (int i = 0; i < numFloats; i++) {
                int offset = i * Float.BYTES;
                floatArray[i] = byteArrayToFloat(data, offset);
            }

            // Print the received float array
            System.out.println("Received float acrray:");
            for (float value : floatArray) {
                System.out.println(value);
            }
        }
    }

    private static float byteArrayToFloat(byte[] bytes, int offset) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes, offset, Float.BYTES);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        return buffer.getFloat();
    }
}
