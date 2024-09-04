package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"net"
)

var udpConn *net.UDPConn

var udpAddress *net.UDPAddr

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
	udpConn = ser

	for {
		p := make([]byte, 28)
		_,remoteaddr,err := ser.ReadFromUDP(p)
		if err != nil {
			fmt.Printf("Some error  %v", err)
			continue
		}
		udpAddress = remoteaddr;

		fmt.Println("read packet", len(p))

		var tagPose [7]float32
		for i := 0; i < len(p)/4; i++ {
			offset := i * 4
			tagPose[i] = math.Float32frombits(binary.LittleEndian.Uint32(p[offset : offset+4]))
		}

		for wsConn, _ := range wsConns {
			wsConn.WriteJSON(tagPose)
		}
	}
}
