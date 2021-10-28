// document.write("<script src='../static/JS/lang.js'></script>")
// setCookie('lan', 'en')
// replace_htmlValue()
// $.ajaxSetup({
//   async: false
// })
let api_url = 'http://192.168.2.92:8002/'
let type = ''
let serialNo = ''
let checkSystemNo = ''
// 用于判断是否将后台返回的response的msg信息显示在内容提示区，存储一个msg信息，直到遇到response里的msg信息与其不同时，才将
// response的msg的信息显示在内容提示区
let messageTip = ''
// 计数器，用于延时执行停止按钮事件，保证后台数据传过来的完整性
let num = 0
// echarts的图表变量
let myChart = null
// 定时器--监听后台系统是否正在运行
let statusSystemInterval = null
let realTimeClData = null
// 控制画图事件何时停止
let stop_time = 10
// 指定图表的配置项和数据
const option = {
  tooltip: {
    formatter: (value) => {
      return `${value.data.x}<br />${value.data.y}`
    }
  },
  dataset: {
    dimensions: ['x', 'y'],
    source: []
  },
  xAxis: {
    name: '时间/s',
    type: 'value'
  },
  yAxis: {
    name: '转速/rpm',
    type: 'value',
    axisLine: {
      onZero: false
    },
    axisLabel: {
      formatter: '{value}'
    }
  },
  visualMap: [
    {
      show: false,
      dimension: 0,
      seriesIndex: 0,
      pieces: []
    }
  ],
  series: [
    {
      type: 'line'
    }
  ]
}

check()

// 校验产品信息和序列号是否为空
function rules() {
  if (
    $('#InputName1').val().trim() === '' ||
    $('#InputName2').val().trim() === ''
  ) {
    $('#exampleModal1').modal({ keyboard: false })
    return 0
  }
  return 1
}
// 获取固定格式的时间
function gettime() {
  var myDate = new Date()
  var year = myDate.getFullYear() //获取当前年
  var mon = myDate.getMonth() + 1 //获取当前月
  var date = myDate.getDate() //获取当前日
  var h = myDate.getHours() //获取当前小时数(0-23)
  var m = myDate.getMinutes() //获取当前分钟数(0-59)
  var s = myDate.getSeconds() //获取当前秒
  var str = year + '-' + mon + '-' + date + ' ' + h + ':' + m + ':' + s
  return str
}
// 日期转时间戳
function getExactTime(time) {
  var date = new Date(time)
  return Date.parse(date)
}

// 获取下拉-输入框的值
function select_options() {
  $.ajax({
    url: 'http://192.168.2.74:8081/api/storage/unqualified/defectList',
    success: function (data, status) {
      // console.log(data, 'defectList')
      if (!data) {
        data['data'] = ['未见异常', '啸叫', '哒哒声', '异常电磁噪音']
      } else {
        data.data.splice(0, 0, '未见异常')
      }
      let str = ''
      for (let i = 0; i < data.data.length; i++) {
        str += `<option value="${data.data[i]}">${data.data[i]}</option>`
      }
      $('#inputvalue').val('未见异常')
      $('#selectvalue').html(str)
      // console.log(arguments)
      // console.log(data)
      // data = data.trim(' ', '')
      // data = data.split('	')
      // let str = ''
      // data[0] = data[0].substring(1)
      // console.log(data[data.length - 1].length - 1)
      // console.log(data[data.length - 1].substring(0, data[data.length - 1].length -1))
      // data[data.length - 1] = data[data.length - 1].substring(0, data[data.length - 1].length - 1)
      // for(let i = 0; i < data.length; i++){
      //   str += `<option value="${data[i].trim('"')}">${data[i].trim('"')}</option>`
      // }
      // $('#selectvalue').html(str)
    },
    error: function (XMLHttpRequest, textStatus, errorThrown) {
      //TODO: 处理status， http status code，超时 408
      // 注意：如果发生了错误，错误信息（第二个参数）除了得到null之外，还可能
      //是"timeout", "error", "notmodified" 和 "parsererror"。
      console.log('err:' + textStatus)
    }
  })
}

function getConfig() {
  $.ajax({
    //请求方式为get
    type: 'GET',
    //json文件位置
    url: 'static/config/config.json',
    //返回数据格式为json
    dataType: 'json',
    //请求成功完成后要执行的方法
    success: function (data) {
      // console.log(data, 'datadatadatadata')
      checkSystemNo = data.SystemNo
      stop_time = data.stopTime
      api_url = data.IP_Address
    }
  })
}
getConfig()
select_options()

// 将select的值赋给input框
$('#selectvalue').on('change', function () {
  // console.log(this.value, 'this.value')
  $('#inputvalue').val(this.value)
})
// 现场判定数据保存事件
$('#saveResult').on('click', function () {
  // save()
})

$('#saveResult').hide()

let visualMap_pieces = []
function check() {
  $('#tipBody').text('')
   realTimeClData = setInterval(() => {
     $.ajax({
    url: api_url + 'QDAQRemoteControl/Command?Cmd=5',
    success: function (result) {
      // console.log(result.code)
        let data = result
        if (Number(data.code) === 3000) {
          if (messageTip !== data.msg) {
            messageTip = data.msg
            type = data.data.type
            serialNo = data.data.serialNo
            $('#InputName1').val(type)
            $('#InputName2').val(serialNo)
            var str =
              gettime() +
              '， ' +
              '产品类型:' +
              type +
              '，' +
              'SN：' +
              serialNo +
              '，' +
              '错误信息：' +
              data.msg
            // 提示信息
            $('#tipBody').append(str + '</br>')
            // $('#resultMsg').html()
            return
          }
        } 
        if (Number(data.code) === 0) {
          if (messageTip !== data.msg) {
            messageTip = data.msg
            // 组装提示信息
            var str =
              gettime() + '，' +
              messageTip
//              '，检测未开始'
            // 提示信息
            $('#tipBody').append(str + '</br>')
          }
          visualMap_pieces = []
          option.dataset.source = []
          option.visualMap[0].pieces = []
          return
        } 
        if (data.data !== null) {
          type = data.data.type
          serialNo = data.data.serialNo
          $('#InputName1').val(type)
          $('#InputName2').val(serialNo)
          if (data.code) {
            console.log(data.data)
            myChart = echarts.init(document.getElementById('idCanvas'))
                // 当返回的数据不为null且测试段，即testName数组为0时，将数据线变成蓝色
                if ((data.data.testName.length === 0 ||
                    option.visualMap[0].pieces.length === 0)
                ) {
                  option.visualMap[0].pieces = [
                    { gte: 0, lt: data.data.x, color: 'blue' }
                  ]
                }
                // 当返回的数据不为null且测试段，即testName数组为1时，将数据线是测试段的段变成红色，其它的为蓝色
                if ((data.data.testName.length === 1 &&
                    option.visualMap[0].pieces.length > 0)
                ) {
                visualMap_pieces = []
                  visualMap_pieces.push({
                    gte: 0,
                    lt: data.data.startX[0],
                    color: 'blue'
                  })
                  visualMap_pieces.push({
                    gte: data.data.startX[0],
                    lt: data.data.endX[0],
                    color: 'red'
                  })
                  visualMap_pieces.push({
                    gte: data.data.endX[0],
                    lt: data.data.x,
                    color: 'blue'
                  })
                  option.visualMap[0].pieces = visualMap_pieces
                }

                // 当返回的数据不为null且测试段，即testName数组大于1时，将数据线是测试段的段变成红色，其它的为蓝色
                if (data.data.testName.length > 1 &&
                  (data.data.testName.length >
                    (option.visualMap[0].pieces.length - 1) / 2)
                ) {
                  for (
                    let i = (option.visualMap[0].pieces.length - 1) / 2;
                    i < data.data.testName.length;
                    i++
                  ) {
                    option.visualMap[0].pieces.splice(-1, 0, {
                      gte: data.data.endX[i - 1],
                      lt: data.data.startX[i],
                      color: 'blue'
                    })
                    option.visualMap[0].pieces.splice(-1, 0, {
                      gte: data.data.startX[i],
                      lt: data.data.endX[i],
                      color: 'red'
                    })
                  }
                }
                // 用于保证最后一段的数据线一定为蓝色
                if (data.data.testName.length > 1) {
                  option.visualMap[0].pieces.splice(-1, 1, {
                    gte: data.data.endX[data.data.endX.length - 1],
                    lt: data.data.x,
                    color: 'blue'
                  })
                }
                if (data.code === 5) {
                  // 当code=5时，表示这次检测已经完成
                  // 后台返回的“-1”表示界限值缺失，“1”表示通过，“0”表示不通过,”2“表示异常
                  const textResult = ['界限值缺失', '不通过', '通过', '异常']
                  const src = [
                    '../static/SVG/wenhao.svg',
                    '../static/SVG/chahao.svg',
                    '../static/SVG/hege.svg',
                    '../static/SVG/yichang.svg'
                  ]
                  let report = ''
                  if (data.data.reportPath && JSON.parse(data.data.reportPath).data) {
                    report = JSON.parse(data.data.reportPath)
                    report = `<a href="${report.data}" target="_blank">${report.data}</a>`
                  } else {
                    report = '报告发送失败'
                  }
                  if (messageTip !== data.msg) {
                    messageTip = data.msg
                    var str =
                      gettime() +
                      '， ' +
                      '产品类型:' +
                      type +
                      '，' +
                      'SN：' +
                      serialNo +
                      '，' +
                      '检测结果：' +
                      textResult[data.result + 1]
                     // +
                     // '，详情请查看报告：' +
                     // report
                    // 提示信息
                    $('#tipBody').append(str + '</br>')
                    // $('#resultmsg').innerText = textResult[data.data.testResult + 1]
                    // $('#result').css('background-color', backgroundColor[data.data.testResult + 1])
                    $('#result').attr('src', src[data.result + 1])
                    $('#resultMsg').html(textResult[data.result + 1])
                  }
                } else if (messageTip !== result.msg) {
                  // 更新提示内容区
                  messageTip = result.msg
                  var str =
                    gettime() +
                    '， ' +
                    '产品类型:' +
                    type +
                    '，' +
                    'SN：' +
                    serialNo +
                    '，' +
                    '信息内容：' +
                    data.msg
                  // 提示信息
                  $('#tipBody').append(str + '</br>')
                  // $('#resultmsg').innerText = '检测中'
                  // $('#result').css('background-color', 'grey')
                }
                option.dataset.source.push({ x: data.data.x, y: data.data.y })
                // 使用刚指定的配置项和数据显示图表。
                myChart.setOption(option)
              }
        }
    },error: function (XMLHttpRequest, textStatus, errorThrown) {
        //TODO: 处理status， http status code，超时 408
        // 注意：如果发生了错误，错误信息（第二个参数）除了得到null之外，还可能
        //是"timeout", "error", "notmodified" 和 "parsererror"。
        console.log('err:' + textStatus)
      }
  })
}, 500)
}


// setTimeout(() => {
//   test = setInterval(() => {
//     // clearInterval(realTimeClData)
//     $.ajax({
//       url: api_url + '/AQS2RTRemoteControl/Command?Cmd=4',
//       success: function (result) {
//         $.ajax({
//           url:
//             api_url +
//             'AQS2RTRemoteControl/Command?Cmd=1&Prop=1&PropNames=Type;SerialNo&PropValues=test;1234',
//           success: function (result) {},
//           error: function (XMLHttpRequest, textStatus, errorThrown) {
//             //TODO: 处理status， http status code，超时 408
//             // 注意：如果发生了错误，错误信息（第二个参数）除了得到null之外，还可能
//             //是"timeout", "error", "notmodified" 和 "parsererror"。
//             console.log('err:' + textStatus)
//           }
//         })
//       },
//       error: function (XMLHttpRequest, textStatus, errorThrown) {
//         //TODO: 处理status， http status code，超时 408
//         // 注意：如果发生了错误，错误信息（第二个参数）除了得到null之外，还可能
//         //是"timeout", "error", "notmodified" 和 "parsererror"。
//         console.log('err:' + textStatus)
//       }
//     })
//   }, 60000)
// }, 1000);