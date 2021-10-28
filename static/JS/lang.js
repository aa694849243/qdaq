//写入cookie函数
function setCookie(name, value) {
  const Days = 30
  let exp = new Date()
  exp.setTime(exp.getTime() + Days * 24 * 60 * 60 * 1000)
  document.cookie = name + '=' + escape(value) + ';expires=' + exp.toGMTString()
}

//获取cookie
function getCookie(name) {
  let arr,
    reg = new RegExp('(^| )' + name + '=([^;]*)(;|$)')
  if ((arr = document.cookie.match(reg))) return unescape(arr[2])
  else return null
}

// 替换掉html文件的文字
function replace_htmlValue(){
  $('[set-lan]').each(function(){
    var me = $(this);
    var a = me.attr('set-lan').split(':');
    var p = a[0];   //文字放置位置
    var m = a[1];   //文字的标识

    //用户选择语言后保存在cookie中，这里读取cookie中的语言版本
    var lan = getCookie('lan');
    console.log(lan)
    //选取语言文字
    switch(lan){
        case 'cn':
            var t = cn[m];  //这里cn[m]中的cn是上面定义的json字符串的变量名，m是json中的键，用此方式读取到json中的值
            break;
        case 'en':
            var t = en[m];
            break;
        default:
            var t = hk[m];
    }

    //如果所选语言的json中没有此内容就选取其他语言显示
    if(t==undefined) t = cn[m];
    if(t==undefined) t = en[m];
    if(t==undefined) t = hk[m];

    if(t==undefined) return true;   //如果还是没有就跳出

    //文字放置位置有（html,val等，可以自己添加）
    switch(p){
        case 'html':
            me.html(t);
            break;
        case 'val':
        case 'value':
            me.val(t);
            break;
        default:
            me.html(t);
    }

});
}

// 替换掉js文件中提示内容的文字
function get_lan(m)
{
    //获取文字
    var lan = getCookie('lan');     //语言版本
    //选取语言文字
    switch(lan){
        case 'cn':
            var t = cn[m];
            break;
        case 'hk':
            var t = hk[m];
            break;
        default:
            var t = en[m];
    }

    //如果所选语言的json中没有此内容就选取其他语言显示
    if(t==undefined) t = cn[m];
    if(t==undefined) t = en[m];
    if(t==undefined) t = hk[m];

    if(t==undefined) t = m; //如果还是没有就返回他的标识

    return t;
}


var cn = {
            "type" : "产品类型",
            "serviceNo" : "序列号"
        };

var en = {
            "type" : "Product Type",
            "serviceNo" : "Service No",
            "email" : "Email",
        };



