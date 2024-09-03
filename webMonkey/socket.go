package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"net"
)
func sendResponse(conn *net.UDPConn, addr *net.UDPAddr) {
    _,err := conn.WriteToUDP([]byte("From server: Hello I got your message "), addr)
    if err != nil {
        fmt.Printf("Couldn't send response %v", err)
    }
}


func startSocket() {
	addr := net.UDPAddr{
		Port: 9001,
		IP:   net.ParseIP("127.0.0.1"),
	}

	ser, err := net.ListenUDP("udp", &addr)
	if err != nil {
		fmt.Printf("Some error %v\n", err)
		return
	}

	for {
		p := make([]byte, 28)
		_,remoteaddr,err := ser.ReadFromUDP(p)
		if err != nil {
			fmt.Printf("Some error  %v", err)
			continue
		}

		fmt.Println("read packet", len(p))

		var tagPose [7]float32
		for i := 0; i < len(p)/4; i++ {
			offset := i * 4
			tagPose[i] = math.Float32frombits(binary.LittleEndian.Uint32(p[offset : offset+4]))
		}

		for wsConn, _ := range wsConns {
			wsConn.WriteJSON(tagPose)
		}
		go sendResponse(ser, remoteaddr)
	}
}
