
var flag_start = false

// =====================================================================================================================================
// =====================================================================================================================================
// 发送数据函数
// xmlHttp 方式发送
function SendData(){

   if (flag_start == true){
      var xmlHttp
      if (window.XMLHttpRequest) {
         // IE7+, Firefox, Chrome, Opera, Safari : Browser Code
         xmlHttp = new XMLHttpRequest()
      }
      else {
         // IE6, IE5 Browser Code
         xmlHttp = new ActiveXObject("Microsoft.XMLHTTP")
      }
      xmlHttp.onreadystatechange = function () {
         if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
            // document.getElementById("data1").innerHTML = xmlHttp.responseText
            // ====================================================================================
            console.log('data2 xmlHttp.responseText = ', xmlHttp.responseText)
            // ====================================================================================
            // ==========================================================
            Draw(xmlHttp.responseText)
            // ==========================================================
         }
      }
      xmlHttp.open("GET", "http://192.168.2.103:9999/data2", true)
      xmlHttp.send()
   }

  
}

// =====================================================================================================================================
// =====================================================================================================================================
// 画图主函数
// 通过 echart 与 canvas 画图
function Draw(xmlData){
// function Draw(){

   var flag_height = false

   if (xmlData != ""){

      // var BoxHeight = document.getElementById('id_ShowBox').offsetHeight
      // console.log('BoxHeight = ', BoxHeight)
      // document.getElementById('idCanvas').style.height = '2000px'

      var obj_JSON = JSON.parse(xmlData)

      // var titlesA = ['chart1', 'chart2']
      var ChannelNum = obj_JSON[0]
      var XInterval = obj_JSON[1]
      var titlesA = obj_JSON[2]
      var SonItemSum = (obj_JSON[3][0].length)
      var XArray = []
      var BoxHeight = ChannelNum * 600
      
      if (flag_height == false){
         document.getElementById('idCanvas').style.height = String(BoxHeight) + 'px'
         // document.getElementById('id_ShowBox').style.height = String(ChannelNum * 600 + 100) + 'px'
         flag_height = true
      }

      // =====================================================================================
      var myChart = echarts.init(document.getElementById('idCanvas'));
      // =====================================================================================
     
      for (i = 0; i < SonItemSum; i++){
         XArray.push(i * XInterval)
      }

      var sourceArray = []
      var xAxisArray = []
      var yAxisArray = []
      var gridArray = []
      var titleArray = []
      var seriesArray = []
      var dataZoomA = []



      for (i = 0; i < ChannelNum; i++){
         sourceArray.push(XArray)
         sourceArray.push(obj_JSON[3][i])
         
         xAxisArray.push({ type: 'value', scale: true, name: obj_JSON[2][i][1], gridIndex: i})
         yAxisArray.push({ type: 'value', scale: true, name: obj_JSON[2][i][2], gridIndex: i})

         gridArray.push({
            // top: String(Math.round(i * (100 / ChannelNum) + (100 / ChannelNum) + 10)) + '%',
            // bottom: String(Math.round(100 - (i * (100 / ChannelNum) * 0.8))) + '%',

            // top: String((i * (100 / ChannelNum) + 20)) + '%',

            top: String(i*600 + 60),
            // bottom: String(BoxHeight - (i*800)),
            bottom: String(BoxHeight - i*600 - 500),

            left: '8%',
            right: '15%'
         })

         titleArray.push({
            text: titlesA[i][0],
            left: '50%',
            top: String(i * 600 + 20),
            textAlign: 'center'
         })

         seriesArray.push({
            type: 'line',
            xAxisIndex: i,
            yAxisIndex: i,
            seriesLayoutBy: 'row'
         })

         dataZoomA.push(
            {
               type: 'slider',
               xAxisIndex: i,
               start: 0,
               end: 100,
               top: String((i+1) * 600 - 50),
               filterMode: 'none'
            },
            {
               type: 'inside',
               xAxisIndex: i,
               start: 0,
               end: 100,
               top: String((i + 1) * 600 - 50),
               filterMode: 'none'
            },
            {
               type: 'slider',
               yAxisIndex: i,
               start: 0,
               end: 100,
               filterMode: 'none'
            },
            {
               type: 'inside',
               yAxisIndex: i,
               start: 0,
               end: 100,
               filterMode: 'none'
            }
         )

      }

      
      // =====================================================
      // =====================================================

      
      var tooltipA = [{
         trigger: 'axis',
         axisPointer: {
            type:'cross'
         }
      }]

      var toolboxA = [{
         feature: {
            dataZoom: {
               yAxisIndex: 'none'
            },
            restore: {},
            saveAsImage: {}
         }
      }]
      

      // =====================================================
      // =====================================================


      option = {
         animation: false,
         legend: {},
         tooltip: tooltipA,
         toolbox: toolboxA,
         dataZoom: dataZoomA,
         title: titleArray,
         dataset: {
            source: sourceArray
         },
         xAxis: xAxisArray,
         yAxis: yAxisArray,
         grid: gridArray,
         series: seriesArray
      }


      // Show the chart
      myChart.setOption(option);

      // ===============================================================================================
      // 以下为测试代码

      // console.log('\n')
      // console.log('option.xAxis = ', JSON.stringify(option.xAxis))
      // console.log('option.yAxis = ', JSON.stringify(option.yAxis))
      // console.log('option.series = ', JSON.stringify(option.series))
      // console.log('option.dataset = ', JSON.stringify(option.dataset))

      // var sourceArray = [
      //    [1, 2, 3, 4, 5, 6],
      //    [10, 30, 40, 80, 60, 50],
      //    [0, 2, 3, 4, 5, 6],
      //    [80, 70, 90, 30, 20, 50]
      // ]

      // var xAxisArray = [{
      //       type: 'value',
      //       name: 'grid1X',
      //       gridIndex: 0,
      //    },
      //    {
      //       type: 'value',
      //       name: 'grid2X',
      //       gridIndex: 1
      //    }
      // ]

      // var yAxisArray = [{
      //       name: 'grid1Y',
      //       gridIndex: 0
      //    },
      //    {
      //       name: 'grid2Y',
      //       gridIndex: 1
      //    }
      // ]

      // var titleArray = [{
      //       text: titlesA[0],
      //       left: '50%',
      //       top: '3%',
      //       textAlign: 'center'
      //    },
      //    {
      //       text: titlesA[1],
      //       left: '50%',
      //       top: '53%',
      //       textAlign: 'center'
      //    }

      // ]

      // var seriesArray = [
      //    // Show X axis data when gridIndex == 0 
      //    {
      //       type: 'line',
      //       xAxisIndex: 0,
      //       yAxisIndex: 0,
      //       seriesLayoutBy: 'row'
      //    },

      //    // Show Y axis data when gridIndex == 1 
      //    {
      //       type: 'line',
      //       xAxisIndex: 1,
      //       yAxisIndex: 1,
      //       seriesLayoutBy: 'row'
      //    }
      // ]

      // option = {
      //    legend: {},
      //    tooltip: {},
      //    title: titleArray,
      //    dataset: {
      //       source: sourceArray
      //    },
      //    xAxis: xAxisArray,
      //    yAxis: yAxisArray,
      //    grid: gridArray,
      //    series: seriesArray
      // }


      // // Show the chart
      // myChart.setOption(option);



   }


   
}



// =====================================================================================================================================
// =====================================================================================================================================
// =====================================================================================================================================
window.setInterval(SendData, 100)
// =====================================================================================================================================
// =====================================================================================================================================
// =====================================================================================================================================


// =======================================================
// =======================================================
function StartBtn(){

   flag_start = true

   console.log('StartBtn()')


   var xmlHttp
   if (window.XMLHttpRequest) {
      // IE7+, Firefox, Chrome, Opera, Safari : Browser Code
      xmlHttp = new XMLHttpRequest()
   }
   else {
      // IE6, IE5 Browser Code
      xmlHttp = new ActiveXObject("Microsoft.XMLHTTP")
   }
   xmlHttp.onreadystatechange = function () {
      if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {

         // document.getElementById("data1").innerHTML = xmlHttp.responseText

         // ====================================================================================
         console.log('start xmlHttp.responseText = ', xmlHttp.responseText)
         // ====================================================================================

         // ==========================================================
         // Draw(xmlHttp.responseText)
         // ==========================================================



      }
   }
   xmlHttp.open("GET", "http://192.168.2.103:9999/start", true)
   xmlHttp.send()
}



// =======================================================
function StopBtn() {

   flag_start = false

   console.log('StopBtn()')

   var xmlHttp
   if (window.XMLHttpRequest) {
      // IE7+, Firefox, Chrome, Opera, Safari : Browser Code
      xmlHttp = new XMLHttpRequest()
   }
   else {
      // IE6, IE5 Browser Code
      xmlHttp = new ActiveXObject("Microsoft.XMLHTTP")
   }
   xmlHttp.onreadystatechange = function () {
      if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {

         // document.getElementById("data1").innerHTML = xmlHttp.responseText

         // ====================================================================================
         console.log('stop xmlHttp.responseText = ', xmlHttp.responseText)
         // ====================================================================================

         // ==========================================================
         // Draw(xmlHttp.responseText)
         // ==========================================================



      }
   }
   xmlHttp.open("GET", "http://192.168.2.103:9999/stop", true)
   xmlHttp.send()
}


window.onload = function(){

   
   


   // Draw()


   // var c = document.getElementById("idCanvas1")
   // var ctx = c.getContext("2d")
   
   // ctx.fillStyle = "#FF0000"
   // ctx.fillRect(15, 10, 270, 1)
   // ctx.fillRect(15, 10, 1, 130)
   
   // ctx.moveTo(15, 140)
   // ctx.lineTo(15, 12)
   // ctx.lineTo(280, 12)
   // ctx.lineWidth = 2
   // ctx.strokeStyle = "red"
   // ctx.stroke()


   // // 基于准备好的dom，初始化echarts实例
   // var myChart = echarts.init(document.getElementById('idCanvas1'));

   // var A = [820, 932, 901, 934, 1290, 1330, 1320]

   // // 指定图表的配置项和数据
   // option = {
   //    xAxis: {
   //       type: 'category',
   //       data: ['Mon', '好人', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
   //    },
   //    yAxis: {
   //       type: 'value'
   //    },
   //    series: [{
   //       type: 'line',
   //       data: A
   //    }]
   // };

   // // 使用刚指定的配置项和数据显示图表。
   // myChart.setOption(option);
}

