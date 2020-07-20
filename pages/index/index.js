//index.js
//获取应用实例
const posenet = require('@tensorflow-models/posenet')
const regeneratorRuntime = require('regenerator-runtime')
const tf = require('@tensorflow/tfjs-core')

Page({
  async onReady() {
    const camera = wx.createCameraContext(this)
    this.canvas = wx.createCanvasContext('pose', this)
    this.loadPosenet()

    let count = 0
    const listener = camera.onCameraFrame((frame) => {
      count++
      if (count === 10) {
        if (this.net) {
          // 只有网络加载成功之后
          // this.pose = this.detectPose(frame,this.net)
          this.drawPose(frame)
        }
        count = 0
      }
    })
    listener.start()
  },
  async loadPosenet() {
    const POSENET_URL =
      'https://www.gstaticcnapps.cn/tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json';
    this.net = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: 193,
      multiplier: 0.5,
      modelUrl: POSENET_URL
    })

    console.log(this.net)

  },
  async detectPose(frame, net) {
    // 4通道转为3通道
    const imgData = {
      data: new Uint8Array(frame.data),
      width: frame.width,
      height: frame.height
    }
    // 用来及时释放内存
    const imgSlice = tf.tidy(() => {
      const imgTensor = tf.browser.fromPixels(imgData, 4)
      return imgTensor.slice([0, 0, 0], [-1, -1, 3])
    })
    const pose = await net.estimateSinglePose(imgSlice, {
      flipHorizontal: false
    })
    console.log(pose)
    // 释放tensor张量
    imgSlice.dispose()
    return pose

  },
  async drawPose(frame) {
    const pose = await this.detectPose(frame, this.net)
    if (pose == null || this.canvas == null)     return
    if (pose.score >= 0.3) {
      // draw circles
      for (i in pose.keypoints) {
        // console.log(pose.keypoints[i])
        const point = pose.keypoints[i]
        if (point.score >= 0.5) {
          const { y, x } = point.position
          this.drawCircle(this.canvas, x, y)
        }
      }
      // this.canvas.draw()
      // draw lines
      const adjacentKeyPoints = posenet.getAdjacentKeyPoints(pose.keypoints, 0.3)
      for (i in adjacentKeyPoints) {
        const points = adjacentKeyPoints[i]
        this.drawLine(this.canvas,points[0], points[1])
      }
      this.canvas.draw()
    }
  },
  drawCircle(canvas, x, y) {
    canvas.beginPath()
    canvas.arc(x * 0.72, y * 0.72, 3, 0, 2 * Math.PI)
    canvas.fillStyle = 'aqua'
    canvas.fill()
  },
  drawLine(canvas, pos0, pos1) {
    canvas.beginPath()
    canvas.moveTo(pos0.x * 0.72, pos0.y * 0.72)
    canvas.lineTo(pos1.x * 0.72, pos1.y * 0.72)
    canvas.lineWidth = 2
    canvas.strokeStyle = `aqua`
    canvas.stroke()
  }

})