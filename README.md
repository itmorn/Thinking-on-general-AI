# Thinking-on-general-AI
通往通用人工智能的思考（天马行空的手稿）

# 为什么有此文
最近看了一些论文，看到很多科学家在一些数据集上使用各种trick，
试图提升算法在该数据集上的表现，感觉有点疲倦了。
诚然，现在的机器学习模型已经帮我们有效地解决了很多问题。
但是，大部分的功效还是处于提升效率的层面，而在精度层面，
还有很长的路要走。就比如说，CV圈非常火的COCO数据集，
当前（2022年8月13日）最好的模型也就是60多的mAP，这个距离
人类的识别水平还有很大距离。不过在ImageNet上的分类效果，似乎
已经超越了人的水平，但是，图像分类实在是一个过于简单的任务了。
如果现在的模型的目标检测的效果能像图像分类一样超越人类的水平，
那这对于提升社会生产力将会有巨大的提升，甚至可以说是一次新的
工业革命。

但是，我觉得很难。我的理由是现在的模型太过于依赖统计，当然，
不是说统计不好，我们在做一件事情的时候，也会梳理以往的经验，
其实也是统计。但是，我们在统计的同时，还会加上当前时刻的特殊性，
因为世界上没有两片一样的叶子，没有哪个场景是完全和之前的场景
一致。人脑在决策时候，不仅会考虑到以往的经验，还会关注到当前场景
的特殊性，再从脑海中调取该特殊性背后潜在可能推理出来的信息，最终
才能做出决策。

虽然很多科学家在统计学习的路上马不停蹄的耕耘，但是，一直给我一种
一眼能看到头的感觉，尤其是在我前两年读了朱松纯的几篇文章之后。
也更加认识到如果不能训练一个具备通用智慧的AI模型，仅在像COCO
数据集上卷，对未来的AI的发展贡献不会太大。

当然，这里我也要自我批评一下，我自己在AI界就是一个无名小卒，如此
大言不惭地对这么大的话题指手画脚多少是有一点班门弄斧了，大家
就全当做是童言无忌吧。

我就是想到哪写到哪，偶尔也会调整一下章节的顺序，就天马行空的想，
就跟进行一次长途旅行一样，走走停停，笔法上我也就没那么注意了，
随性一点。

# 通用智能 之于我
小时候喜欢看《终结者》，尤其痴迷于机器人的第一视角，可以看到很多
酷炫的参数在闪烁，让我激动不已，经常拉着小伙伴来我家里看，当然，
还有《机器人之恋》我也很喜欢。这里面的机器人就是一种具有通用智慧
的机器人，比如七弟会打拳，开车，打枪，帮助女主作弊，以及最后黑化
爱上女主。

好像是2015年的时候，我那时候大学三年级，是一个风和日丽的下午，
我正要去上一节电子商务的课程，然后就打不起精神，因为昨天刚写了
很多代码，当时在琢磨着开发一个C++的游戏，写得我头晕脑胀的。然后，
看了一会alphaGo和李世石的比赛，我当时的感觉就是这玩意有点意思，
不过结果应该是没有悬念的，肯定是计算机会赢了，不然他们拉出来比赛
输了那不闹呢吗。有意思的是当时有一个美女主持人问叨叨魏，谁会赢，
叨叨魏笑着说，肯定是AlphaGo了。然后美女又问，你说他这么厉害，
会不会成为取代人类的生物。然后叨叨魏就反问她，你先说一下生物的定义。
然后，反正又是噼里啪啦说了一顿，我印象比较深的就是，有人问道
AlphaGo和人类的思考方式的时候，叨叨魏说了句，它不具有像人类一样的
通用智慧。

最近一次通用智慧对我触动比较大是在我前领导去了BIGAI之后的一次聚餐上，
饭桌上聊到了朱松纯的一些想法，后来我自己也看了他的一些采访记录，比如
[《浅谈人工智能：现状、任务、构架与统一 | 正本清源》](https://mp.weixin.qq.com/s/-wSYLu-XvOrsST8_KEUa-Q)
。突然感觉这可能是一种不错的做法，尤其是里面有一句 “Go Dark， Beyond Deep”
大概意思就是说，一张图片里面蕴含的信息是非常少的，有很多像宇宙中的暗物质
一样的东西在图像中是看不到的，其实就是用暗物质比喻人的常识了。其实，
这也很好理解，假设我们看一张图片，一个远处的人，拿着一个不知道什么的东西
放在耳边，嬉笑的说话，常识就会告诉我们他手里应该拿的是手机，他可能正在
打电话。我就思考，如果能把这些common sense建模起来是不是就牛皮了。

最近公司正在做一个项目，需要识别网考学生是否在作弊，要识别的物体有使用
多人、看手机、看屏幕、带耳机、看笔记本、看笔记纸之类的。这个项目乍一看挺简单是吧？不就是个
目标检测吗。等我做了一段时间就发现不是那么回事，首先是没有标注数据，
给我的数据都是一整场考试下来的30多w的抓拍照，像我这么懒的人是不可能
去标注数据的，而且这个工作量也不能接受。只能是找找公开数据集，然后就找
到了旷视的objects365，这里面还算不错，基本类别都有，只是没有笔记纸，
得，那就不识别笔记纸了呗~ 先别急，这里面还有很多别的问题了，我们再说
笔记本，objects365里面有一个book类，应该是和笔记本最像的了，但是
很多人家里的背景就是一个书柜，里面自然就有很多书，得，笔记本这一类也
有毛病。您还是先别急，再看多人，很多考生的海报里的人，哪怕是一个布娃娃
在objects365上都是会标记为人的，我哭了，人也凉了。我刚擦完眼泪，再看
耳机，objects365中的耳机有头戴式和入耳式，头戴式的还算简单，但是入耳式
就难搞了啊，有的人比较胖，脸都把耳机挡住了，但是我们能清楚的看到两根耳机线
但就是无能为力，因为objects365标注入耳式耳机的时候，只标注了耳机的位置，
不算耳机线的，得，耳机也凉了一半。我刚冠上一个暖水袋准备捂一捂拔凉的内心的时候，
突然想到，什么叫TM的“看手机”、“看屏幕”啊？这意思是检测到手机还不行，还得是
有看的动作。我吐了，这个项目可以说是我职业生涯里对我非常大的打击。
我突然就想到了《香水》里的男主，当得知不能提取味道之后大病一场的场景了。

至此，我认识到 通用AI能力的重要性。没有它的话，未来我一定会很难受的。
因为我这个人稍微有点强迫症，感觉这个工程被自己做成了一个粑粑活。所以我就
准备思考如何赋予机器通用的智能。

# 婴儿成长的启示
[婴儿成长的启示](婴儿成长的启示/README.md)

# 摄像机
## 单目
优点：
- 简单
- 视频、图片都可以作为训练数据

缺点：
- 感知深度能力差，建模会有误差

## 多目
可以从多角度辅助构建场景，但是需要有高效的算法支持


# 物体
婴儿认识物体也是一个一个认识的，那么我们就让计算机像婴儿一样来学习。
物体会有自己的属性，且在一定范围内会变化。
当一个物体仅露出一小部分的时候，也能识别。物体都是由更基础的基础单元组成的。
比如收集，只露出扬声器部分，那么就可以先识别出扬声器，然后按照图谱搜索
会用到扬声器的电器。

## 建模
使用unit3d建模，然后把物体的形态模拟出来。
36*36种形态采集模型特征
待测图像多尺度分析，类似人看东西一样，先宏观看，然后再看细节。

## 观察
变换角度观察物体，远近，上下

## 存储
将图像向量化是一个核心。
存储时还要考虑到物体的空间关系


## 学习和自我建模
看更多视频，学习该物体的更多形态，颜色，纹理，质地信息
可能有的还不认识，但是可以告诉计算机某个物体是什么

## 物体困惑度


# 物体联系
学习物体之间的联系

## 联系困惑度
















