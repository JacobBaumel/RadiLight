package main

import (
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

var wsConns = make(map[*websocket.Conn]struct{})

func startServer() {
	upgrader := websocket.Upgrader{} // use default options

	r := gin.Default()
	r.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "pong",
		})
	})

	r.GET("/ws", func(ctx *gin.Context) {
		c, err := upgrader.Upgrade(ctx.Writer, ctx.Request, nil)
		if err != nil {
			log.Print("upgrade:", err)
			return
		}

		wsConns[c] = struct{}{}
		c.SetCloseHandler(func(code int, text string) error {
			delete(wsConns, c)
			return nil
		})

		for {
			_, message, err := c.ReadMessage()
			if err != nil {
				log.Println("read:", err)
				break
			}
			log.Printf("recv: %s", message)
			_,err = udpConn.WriteToUDP(message, udpAddress);
			if err != nil {
				log.Println("write:", err)
				break
			}
		}
	})

	r.GET("/", func(ctx *gin.Context) {
		ctx.File("index.html")
	})

	r.Run() // listen and serve on 0.0.0.0:8080 (for windows "localhost:8080")
}
