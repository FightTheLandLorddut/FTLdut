# 叫分斗地主（FightThelandlord2）样例程序
# 无脑策略
# 作者：zhouys
# 游戏信息：http://www.botzone.org/games#FightThelandlord2

import os
import random
import math
import json
from collections import Counter


# 红桃 方块 黑桃 草花
# 3 4 5 6 7 8 9 10 J Q K A 2 joker & Joker
# (0-h3 1-d3 2-s3 3-c3) (4-h4 5-d4 6-s4 7-c4) …… 52-joker->16 53-Joker->17

def evalAndBid(poker, maxScore):
    evals = [i for i in range(maxScore + 1, 4)]
    evals += [0]
    return random.choice(evals)


def printBid(poker, bidHistory):
    bidHistory += [0]
    maxScore = max(bidHistory)
    if len(full_input["requests"]) == 1:
        print(json.dumps({
            "response": str(evalAndBid(poker, maxScore))
        }))


def getPointFromSerial(x):
    return int(x / 4) + 3 + (x == 53)


def convertToPointList(poker):
    return [getPointFromSerial(i) for i in poker]


_online = os.environ.get("USER", "") == "root"

if _online:
    full_input = json.loads(input())
else:
    with open("botlogs.json") as fo:
        full_input = json.load(fo)

bid_info = full_input["requests"][0]
if "bid" in bid_info and len(full_input["requests"]) == 1:
    printBid(bid_info["own"], bid_info["bid"])
    exit(0)

if "bid" in bid_info and len(full_input["requests"]) > 1:
    user_info = full_input["requests"][1]
    my_history = full_input["responses"][1:]
    others_history = full_input["requests"][2:]
else:
    user_info = bid_info
    my_history = full_input["responses"]
    others_history = full_input["requests"][1:]

# 手牌，公共牌，地主位置，当前我的bot位置，最终叫牌底分
poker, publiccard, landlordPos, currBotPos, finalBid = user_info["own"], \
    user_info["publiccard"], user_info["landlord"], user_info["pos"], user_info["finalbid"]
if landlordPos == currBotPos:
    poker.extend(publiccard)

history = user_info["history"]
last_history = full_input["requests"][-1]["history"]

start = (landlordPos + 3 - currBotPos) % 3
history = history[(2 - start):]

for i in range(len(my_history)):
    history.append(my_history[i])
    history.append(others_history[i]["history"])

lenHistory = len(history)

for tmp in my_history:
    for j in tmp:
        poker.remove(j)
poker.sort()  # 用0-53编号的牌


def separateHands(poker):  # 拆分手牌牌型并组成基本牌集合
    res = []
    if len(poker) == 0:
        return res
    serialPoker = [i for i in poker]  # 序数牌，范围在[0,53]之间
    pointPoker = convertToPointList(serialPoker)  # 点数牌，范围在[3,17]之间
    if 16 in pointPoker and 17 in pointPoker:  # 单独列出火箭/小王、大王
        pointPoker = pointPoker[:-2]
        serialPoker = serialPoker[:-2]
        res += [[16, 17]]
    elif 16 in pointPoker or 17 in pointPoker:
        res += [[pointPoker[-1]]]
        pointPoker = pointPoker[:-1]
        serialPoker = serialPoker[:-1]

    singlePointPoker = list(set(pointPoker))  # 都有哪些牌
    singlePointPoker.sort()

    for i in singlePointPoker:  # 分出炸弹，其实也可以不分，优化点之一
        if pointPoker.count(i) == 4:
            idx = pointPoker.index(i)
            res += [serialPoker[idx:idx + 4]]
            pointPoker = pointPoker[0:idx] + pointPoker[idx + 4:]
            serialPoker = serialPoker[0:idx] + serialPoker[idx + 4:]

    # 为了简便处理带2的情形，先把2单独提出来
    specialCnt, specialRes = 0, []
    if 15 in pointPoker:
        specialCnt = pointPoker.count(15)
        idx = pointPoker.index(15)
        specialRes = [15] * specialCnt
        pointPoker, serialPoker = pointPoker[:-specialCnt], serialPoker[:-specialCnt]

    def findLongestPointSeqList(pointPoker, dupTime, minLen):  # 这里的pointPoker是点数牌，找最长的顺子，返回值为牌型组合
        resSeq, tmpSeq = [], []
        singlePointPoker = list(set(pointPoker))
        singlePointPoker.sort()
        for currPoker in singlePointPoker:
            if pointPoker.count(currPoker) >= dupTime:
                if len(tmpSeq) == 0:
                    tmpSeq = [currPoker]
                    continue
                elif currPoker == (tmpSeq[-1] + 1):
                    tmpSeq += [currPoker]
                    continue
            if len(tmpSeq) >= minLen:
                tmpSeq = [i for i in tmpSeq for j in range(dupTime)]
                resSeq += [tmpSeq]
            tmpSeq = []
        return resSeq

    # 一定保证rmPointPoker是pointPoker的子集
    def removeSeqPokerFromHands(serialPoker, pointPoker, rmPointPoker):
        singleP = list(set(rmPointPoker))
        singleP.sort()
        for currPoker in singleP:
            idx = pointPoker.index(currPoker)
            cntPoker = rmPointPoker.count(currPoker)
            pointPoker = pointPoker[0:idx] + pointPoker[idx + cntPoker:]
            serialPoker = serialPoker[0:idx] + serialPoker[idx + cntPoker:]
        return serialPoker, pointPoker

    # 单顺：1，5；双顺：2，3；飞机：3，2；航天飞机：4，2。因为前面已经把炸弹全都提取出来，所以这里就不主动出航天飞机了
    validParaIdx, paraList = [0, 1, 2], [[1, 5], [2, 3], [3, 2]]
    allSeq = [[], [], []]  # 分别表示单顺、双顺、三顺（飞机不带翼）
    while (True):  # serialPoker，这里会找完所有的最长顺子
        if len(pointPoker) == 0 or len(validParaIdx) == 0:
            break
        dupTime = random.choice(validParaIdx)
        selectedPara = paraList[dupTime]
        pointSeqList = findLongestPointSeqList(pointPoker, selectedPara[0], selectedPara[1])
        allSeq[dupTime].extend(pointSeqList)
        for pointSeq in pointSeqList:
            serialPoker, pointPoker = removeSeqPokerFromHands(serialPoker, pointPoker, pointSeq)
        if len(pointSeqList) == 0:
            validParaIdx.remove(dupTime)
    res += allSeq[0] + allSeq[1]  # 对于单顺和双顺没必要去改变
    planeWithoutWings = allSeq[2]

    allRetail = [[], [], []]  # 分别表示单张，对子，三张
    singlePointPoker = list(set(pointPoker))  # 更新目前为止剩下的牌，pointPoker和serialPoker是一一对应的
    singlePointPoker.sort()
    for currPoker in singlePointPoker:
        cntPoker = pointPoker.count(currPoker)
        allRetail[cntPoker - 1].append([currPoker for i in range(cntPoker)])

    # 接下来整合有需要的飞机or三张 <-> 单张、对子。这时候的飞机和三张一定不会和单张、对子有重复。
    # 如果和单张有重复，即为炸弹，而这一步已经在前面检测炸弹时被检测出
    # 如果和对子有重复，则同一点数的牌有5张，超出了4张

    # 先整合飞机
    for triplePokerSeq in planeWithoutWings:
        lenKind = int(len(triplePokerSeq) / 3)
        for t in range(2):  # 分别试探单张和对子的个数是否足够
            retailPoker = allRetail[t]
            if len(retailPoker) >= lenKind:
                triplePokerSeq.extend([i[j] for i in retailPoker[0:lenKind] for j in range(t + 1)])
                allRetail[t] = allRetail[t][lenKind:]
                break
        res.extend([triplePokerSeq])

    if specialCnt == 3:
        allRetail[2] += [specialRes]
    elif specialCnt > 0 and specialCnt <= 2:
        allRetail[specialCnt - 1] += [specialRes]

    # 之后整合三张
    for currPoker in allRetail[2]:
        for t in range(2):
            retailPoker = allRetail[t]
            if len(retailPoker) >= 1:
                currPoker.extend(retailPoker[0])
                allRetail[t] = allRetail[t][1:]
                break
        res.append(currPoker)

    res += allRetail[0] + allRetail[1]
    return res


# J,Q,K,A,2-11,12,13,14,15
# 单张：1 一对：2 三带：零3、一4、二5 单顺：>=5 双顺：>=6
# 四带二：6、8 飞机：>=6
def checkPokerType(poker):  # poker：list，表示一个人出牌的牌型
    poker.sort()
    lenPoker = len(poker)

    ################################################# 0张 #################################################
    if lenPoker == 0:
        return "空", [], []

    ################################################# 1张 #################################################
    if lenPoker == 1:
        return "单张", poker, []

    ################################################# 2张 #################################################
    if lenPoker == 2 and poker == [52, 53]:
        return "火箭", poker, []
    if lenPoker == 2 and getPointFromSerial(poker[0]) == getPointFromSerial(poker[1]):
        return "一对", poker, []
    if lenPoker == 2:
        return "错误", poker, []

    #################################### 转换成点数，剩下牌一定大于等于3张 ###################################
    # 扑克牌点数
    ptrOfPoker = [getPointFromSerial(i) for i in poker]
    firstPtrOfPoker = ptrOfPoker[0]
    # 计数
    cntPoker = Counter(ptrOfPoker)
    keys, vals = list(cntPoker.keys()), list(cntPoker.values())

    ################################################# 4张 #################################################
    if lenPoker == 4 and vals.count(4) == 1:
        return "炸弹", poker, []

    ############################################## >=5张 单顺 #############################################
    singleSeq = [firstPtrOfPoker + i for i in range(lenPoker)]
    if (lenPoker >= 5) and (15 not in singleSeq) and (ptrOfPoker == singleSeq):
        return "单顺", poker, []

    ############################################## >=6张 双顺 #############################################
    pairSeq = [firstPtrOfPoker + i for i in range(int(lenPoker / 2))]
    pairSeq = [j for j in pairSeq for i in range(2)]
    if (lenPoker >= 6) and (lenPoker % 2 == 0) and (15 not in pairSeq) and (ptrOfPoker == pairSeq):
        return "双顺", poker, []

    ################################################# 3张带 ################################################
    if (lenPoker <= 5) and (vals.count(3) == 1):
        if vals.count(1) == 2:
            return "错误", poker, []
        specialPoker = keys[vals.index(3)]
        triplePoker = [i for i in poker if getPointFromSerial(i) == specialPoker]
        restPoker = [i for i in poker if i not in triplePoker]
        tripleNames = ["三带零", "三带一", "三带二"]
        return tripleNames[lenPoker - 3], triplePoker, restPoker

    ############################################## 6张 四带二只 ############################################
    if (lenPoker == 6) and (vals.count(4) == 1) and (vals.count(1) == 2):
        specialPoker = keys[vals.index(4)]
        quadruplePoker = [i for i in poker if getPointFromSerial(i) == specialPoker]
        restPoker = [i for i in poker if i not in quadruplePoker]
        return "四带两只", quadruplePoker, restPoker

        ############################################## 8张 四带二对 ############################################
    if (lenPoker == 8) and (vals.count(4) == 1) and (vals.count(2) == 2):
        specialPoker = keys[vals.index(4)]
        quadruplePoker = [i for i in poker if getPointFromSerial(i) == specialPoker]
        restPoker = [i for i in poker if i not in quadruplePoker]
        return "四带两对", quadruplePoker, restPoker

        # 分别表示张数有0、1、2、3张的是什么点数的牌
    keyList = [[], [], [], [], []]
    for idx in range(len(vals)):
        keyList[vals[idx]] += [keys[idx]]
    lenKeyList = [len(i) for i in keyList]
    ################################################## 飞机 ################################################
    if lenKeyList[3] > 1 and 15 not in keyList[3] and \
            (keyList[3] == [keyList[3][0] + i for i in range(lenKeyList[3])]):
        if lenKeyList[3] * 3 == lenPoker:
            return "飞机不带翼", poker, []
        triplePoker = [i for i in poker if getPointFromSerial(i) in keyList[3]]
        restPoker = [i for i in poker if i not in triplePoker]
        if (lenKeyList[3] == lenKeyList[1]) and (lenKeyList[1] * 4 == lenPoker):
            return "飞机带小翼", triplePoker, restPoker
        if (lenKeyList[3] == lenKeyList[2]) and (lenKeyList[2] * 5 == lenPoker):
            return "飞机带大翼", triplePoker, restPoker

    ################################################# 航天飞机 ##############################################
    if lenKeyList[4] > 1 and lenKeyList[3] == 0 and 15 not in keyList[4] and \
            (keyList[4] == [keyList[4][0] + i for i in range(lenKeyList[4])]):
        if lenKeyList[4] * 4 == lenPoker:
            return "航天飞机不带翼", poker, []
        quadruplePoker = [i for i in poker if getPointFromSerial(i) in keyList[4]]
        restPoker = [i for i in poker if i not in quadruplePoker]
        if (lenKeyList[4] == lenKeyList[1]) and (lenKeyList[1] * 5 == lenPoker):
            return "航天飞机带小翼", quadruplePoker, restPoker
        if (lenKeyList[4] == lenKeyList[2]) and (lenKeyList[2] * 6 == lenPoker):
            return "航天飞机带大翼", quadruplePoker, restPoker

    return "错误", poker, []


def recover(history):  # 只考虑倒数3个，返回最后一个有效牌型及主从牌，且返回之前有几个人选择了pass；id是为了防止某一出牌人在某一牌局后又pass，然后造成连续pass
    lenHistory = len(history)
    typePoker, mainPoker, restPoker, cntPass = "任意牌", [], [], 0

    while (lenHistory > 0):
        lastPoker = history[lenHistory - 1]
        typePoker, mainPoker, restPoker = checkPokerType(lastPoker)
        if typePoker == "空":
            cntPass += 1
            lenHistory -= 1
            continue
        break
    return typePoker, mainPoker, restPoker, cntPass


def searchCard(poker, objType, objMP, objSP):  # 搜索自己有没有大过这些牌的牌
    if objType == "火箭":  # 火箭是最大的牌
        return []
    # poker.sort() # 要求poker是有序的，使得pointPoker一般也是有序的
    pointPoker = convertToPointList(poker)
    singlePointPoker = list(set(pointPoker))  # 都有哪些牌
    singlePointPoker.sort()
    countPoker = [pointPoker.count(i) for i in singlePointPoker]  # 这些牌都有几张

    res = []
    idx = [[i for i in range(len(countPoker)) if countPoker[i] == k] for k in range(5)]  # 分别有1,2,3,4的牌在singlePoker中的下标
    quadPoker = [singlePointPoker[i] for i in idx[4]]
    flag = 0
    if len(poker) >= 2:
        if poker[-2] == 52 and poker[-1] == 53:
            flag = 1

    if objType == "炸弹":
        for currPoker in quadPoker:
            if currPoker > newObjMP[0]:
                res += [[(currPoker - 3) * 4 + j for j in range(4)]]
        if flag:
            res += [[52, 53]]
        return res

    newObjMP, lenObjMP = convertToPointList(objMP), len(objMP)
    singleObjMP = list(set(newObjMP))  # singleObjMP为超过一张的牌的点数
    singleObjMP.sort()
    countObjMP, maxObjMP = newObjMP.count(singleObjMP[0]), singleObjMP[-1]
    # countObjMP虽取首元素在ObjMP中的个数，但所有牌count应相同；countObjMP * len(singleObjMP) == lenObjMP

    newObjSP, lenObjSP = convertToPointList(objSP), len(objSP)  # 只算点数的对方拥有的主牌; 对方拥有的主牌数
    singleObjSP = list(set(newObjSP))
    singleObjSP.sort()
    countObjSP = 0
    if len(objSP) > 0:  # 有可能没有从牌，从牌的可能性为单张或双张
        countObjSP = newObjSP.count(singleObjSP[0])

    tmpMP, tmpSP = [], []

    for j in range(1, 16 - maxObjMP):
        tmpMP, tmpSP = [i + j for i in singleObjMP], []
        if all([pointPoker.count(i) >= countObjMP for i in tmpMP]):  # 找到一个匹配的更大解
            if j == (15 - maxObjMP) and countObjMP != lenObjMP:  # 与顺子有关，则解中不能出现2（15）
                break
            if lenObjSP != 0:
                tmpSP = list(set(singlePointPoker) - set(tmpMP))
                tmpSP.sort()
                tmpSP = [i for i in tmpSP if pointPoker.count(i) >= countObjSP]  # 作为从牌有很多组合方式，是优化点
                species = int(lenObjSP / countObjSP)
                if len(tmpSP) < species:  # 剩余符合从牌特征的牌种数少于目标要求的牌种数，比如334455->lenObjSP=6,countObjSP=2,tmpSP = [8,9]
                    continue
                tmp = [i for i in tmpSP if pointPoker.count(i) == countObjSP]
                if len(tmp) >= species:  # 剩余符合从牌特征的牌种数少于目标要求的牌种数，比如334455->lenObjSP=6,countObjSP=2,tmpSP = [8,9]
                    tmpSP = tmp
                tmpSP = tmpSP[0:species]
            tmpRes = []
            idxMP = [pointPoker.index(i) for i in tmpMP]
            idxMP = [i + j for i in idxMP for j in range(countObjMP)]
            idxSP = [pointPoker.index(i) for i in tmpSP]
            idxSP = [i + j for i in idxSP for j in range(countObjSP)]
            idxAll = idxMP + idxSP
            tmpRes = [poker[i] for i in idxAll]
            res += [tmpRes]

    if objType == "单张":  # 以上情况少了上家出2，本家可出大小王的情况
        if 52 in poker and objMP[0] < 52:
            res += [[52]]
        if 53 in poker:
            res += [[53]]

    for currPoker in quadPoker:  # 把所有炸弹先放进返回解
        res += [[(currPoker - 3) * 4 + j for j in range(4)]]
    if flag:
        res += [[52, 53]]
    return res


lastTypeP, lastMP, lastSP, countPass = recover(last_history)


def randomOut(poker):
    sepRes, res = separateHands(poker), []
    lenRes = len(sepRes)
    idx = random.randint(0, lenRes - 1)
    tmp = sepRes[idx]  # 只包含点数
    pointPoker, singleTmp = convertToPointList(poker), list(set(tmp))
    singleTmp.sort()
    for currPoker in singleTmp:
        tmpCount = tmp.count(currPoker)
        idx = pointPoker.index(currPoker)
        res += [poker[idx + j] for j in range(tmpCount)]
    # tmpCount = pointPoker.count(pointPoker[0])
    # res = [[poker[i] for i in range(tmpCount)]]
    return res


if countPass == 2:  # 长度为0，自己是地主，随便出；在此之前已有两个pass，上一个轮是自己占大头，不能pass，否则出错失败
    # 有单张先出单张
    res = randomOut(poker)
    print(json.dumps({
        "response": res
    }))
    exit(0)

if currBotPos == 1 and countPass == 1:  # 上一轮是农民乙出且地主选择pass，为了不压过队友选择pass
    print(json.dumps({
        "response": []
    }))
    exit()

res = searchCard(poker, lastTypeP, lastMP, lastSP)
lenRes = len(res)

if lenRes == 0:  # 应当输出pass
    print(json.dumps({
        "response": []
    }))
else:
    pokerOut, typeP = [], "空"
    for i in range(lenRes):
        pokerOut = res[i]
        typeP, _, _ = checkPokerType(pokerOut)
        if typeP != "火箭" and typeP != "炸弹":
            break

    if (currBotPos == 2 and countPass == 0) or (currBotPos == 1 and countPass == 1):  # 两个农民不能起内讧，起码不能互相炸
        if typeP == "火箭" or typeP == "炸弹":
            pokerOut = []
    else:  # 其他情况是一定要怼的
        pokerOut = res[0]

    print(json.dumps({
        "response": pokerOut
    }))