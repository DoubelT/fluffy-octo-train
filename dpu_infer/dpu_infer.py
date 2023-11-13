import sys
import xir
import vart
import time
from typing import List
from ctypes import *
import importlib
import numpy as np



data = [
[0.3988674,0.7660978,-0.39003128,-0.58781725],
[-0.24941276,1.0770591,-1.0709577,-0.64423376],
[-0.13359365,0.88524115,-0.8218231,-0.9429486],
[-0.7681015,1.7115785,0.36621013,-1.7803279],
[0.70055956,1.2786709,-1.0972395,-0.1297751],
[1.4749706,-0.8135648,0.26992053,1.148247],
[-0.064673685,0.9851689,-0.11976045,0.14851576],
[1.6637564,0.3632291,-0.8054172,0.76274574],
[1.2536854,-0.22796744,-0.03355438,1.3429006],
[0.18664053,0.6804888,0.18828319,0.59911644],
[0.37731794,0.025352327,-1.0535597,0.50862247],
[-0.360968,0.11340968,-1.0094644,0.13673875],
[0.012770931,-0.41245392,1.3561494,0.74229294],
[-0.4230928,-1.56304,1.150141,0.74043316],
[-0.7605556,0.035128962,1.44314,1.0156276],
[0.42403182,0.4121461,-1.4790289,-0.70870286],
[-0.62262523,-0.65634483,0.30037877,0.85819834],
[-0.6652328,-0.5707359,0.2141727,0.80613416],
[-1.2992215,-2.4568346,3.2419074,0.49932835],
[-0.6005844,-0.7122524,0.70745796,0.7955938],
[0.6987346,1.3041847,-1.0582899,-0.7626268],
[0.5648753,-0.0955471,0.62522066,1.0007542],
[-0.3970896,-0.87577057,0.6030692,1.2883534],
[1.0884846,1.2751756,-1.1858222,-0.029979764],
[-1.7582177,0.27726555,-0.1939009,0.11318476],
[1.2261343,0.7126386,-0.81293947,0.53589547],
[0.9860721,0.7458353,-0.31964502,-0.12608427],
[-0.49768057,-1.4302701,0.9961191,0.829065],
[1.8165327,1.4338138,-1.4554931,-1.519384],
[1.6653357,1.0844043,-1.098024,-0.18494527],
[-0.32409525,0.1748894,-1.1885911,-0.91879064],
[0.4062728,0.081952184,0.39982957,0.65552086],
[0.35980466,-0.31986114,0.61631393,0.8507592],
[-2.3867662,1.0470201,-0.2498541,-0.9764534],
[-1.5277718,0.3820901,-0.40248683,-0.44651246],
[1.1795608,1.0428157,-0.9357184,-0.32998878],
[0.406343,0.067987956,-1.0673811,0.51110536],
[-0.0015169805,-0.46591306,-0.5954629,0.7212171],
[-0.45485896,0.021164758,-0.89538443,0.044986524],
[0.4219611,0.24932022,0.3156079,0.60903317],
[0.37809008,-0.29889673,1.0904243,0.42185035],
[-0.42135903,0.27866703,-1.1417271,-1.0756255],
[-0.18775468,0.0015945781,-0.8593883,0.29540944],
[0.6304713,1.3209518,-1.0659966,-0.73473006],
[-1.0608088,-0.37157267,-0.13221602,0.7732813],
[-0.797302,-1.8401294,1.7845955,0.517303],
[-0.6492988,-0.68604624,0.002418661,0.864396],
[1.105682,-0.94668925,0.44570157,0.5073858],
[0.6791856,-1.4603261,1.5951775,0.35242984],
[0.30301803,0.41808978,-1.4659688,-0.82335716],
[1.2271872,-0.7978445,0.16276287,0.7150194],
[1.6949574,0.36707902,-0.70716625,1.0515673],
[-0.149336,1.1071153,-0.23067693,-1.2615554],
[1.1318291,0.92367226,-0.8714562,0.018959183],
[-0.19393875,0.85518515,-0.13557798,0.24644649],
[-0.0587704,0.10083005,-1.0035342,0.2799129],
[-0.9458317,1.7405876,-0.2812907,-3.1513388],
[1.3562381,0.98062676,-0.89873016,0.106368765],
[1.3870181,-1.1294398,0.42888027,0.5532504],
[0.8544591,1.3831407,-1.0873637,-0.92315245],
[-0.82274723,-1.3457248,0.9457014,0.7999365],
[0.7384642,0.13331755,0.13528125,0.66357833],
[-0.05591352,0.51801753,-1.4829748,-2.2222648],
[0.9815797,1.0012438,-0.4351095,-0.650417],
[-0.784632,1.700755,0.12420551,-1.9600744],
[1.3061551,0.92298,-0.9193126,-0.012628265],
[1.0361202,0.24617954,-0.13795003,0.68775064],
[0.48710078,0.8352772,-0.3455458,0.75654805],
[-0.68021905,-0.2936557,1.2369239,0.324538],
[-0.048230834,0.5194191,-1.5359536,-2.419986],
[-0.7691544,-1.3967357,1.6082375,0.72059876],
[0.6907675,1.2748209,-1.1223445,-1.0396763],
[-0.64206886,-1.9383179,1.83619,0.40015614],
[1.8610706,0.5861501,-0.8692872,0.60345477],
[0.109381914,-0.94250166,0.024362456,1.3224334],
[0.030491317,-0.856555,0.36503333,0.09521969],
[1.3487625,-0.75346965,0.14536478,1.0199654],
[-0.37760037,0.112345904,-1.0274394,0.07475252],
[-0.83815473,-0.27059463,-0.060043823,0.3896206],
[-0.06622145,-0.056761354,-0.67532355,0.2867337],
[-0.29912037,0.18572982,-0.75993747,-0.4905144],
[0.47776502,-0.49421132,0.008741058,0.44230312],
[0.037461534,0.35205096,-1.4592541,-1.5138239],
[0.623487,0.9533737,-0.6543719,-1.0154704],
[0.63843817,-1.1738147,0.6424111,-0.031226043],
[1.267689,1.5812571,-1.353481,-1.7047385],
[0.8505634,0.8625303,-0.5309976,0.79931337],
[-1.0612651,-0.32335475,-0.052544624,1.0664742],
[0.16232195,0.50054115,-1.465369,-2.1125474],
[0.49299702,0.7294058,-0.15396369,0.83216625],
[1.0736036,1.1416966,-1.0476063,0.28363246],
[0.9126146,0.081952184,0.13982692,0.73547214],
[-0.031314168,-0.28457066,-0.41415095,0.6375463],
[-0.06804649,0.047016278,-1.1213523,0.24830145],
[0.52626884,0.71473247,-0.28781384,-0.41674647],
[-1.4116366,-1.4152421,2.0807557,0.48878798],
[-1.6058624,0.16301896,-0.19034283,0.22226937],
[-0.05549236,1.2510632,-1.1061462,-1.7549238],
[0.6888021,1.2999802,-1.0812258,-0.87478864],
[0.7372006,0.18642214,-0.53752303,0.50862247],
[-1.6917442,-1.6102178,2.4131198,-0.1297751],
[1.4484725,-0.71434623,0.26161373,1.3422774],
[-0.012997142,0.8572958,-0.10353914,0.24334525],
[1.0684444,-0.33139217,0.23593189,1.7414106],
[-2.5217838,1.1832852,-0.36215344,-1.2181765],
[1.6743205,0.2874137,-0.6371584,1.0528135],
[1.0866596,-0.1846412,0.10226173,1.4959966],
[0.07410262,0.3300324,-1.3801551,-1.373142],
[-1.3773118,-2.3610945,2.967691,0.27185544],
[-1.2705474,-1.5847039,2.3905761,0.2898301],
[-0.9005568,-0.040686443,-0.36729395,-0.26925844],
[0.16049692,0.07671769,-1.06422,0.5104822],
[0.58273953,1.096984,-0.34258008,-1.0012344],
[-1.1637828,-0.0064428756,-0.18876223,0.34499553],
[-0.3410084,-1.1780022,0.44588614,0.5922985],
[-1.3981243,0.014528784,-0.30481738,-0.6026764],
[1.4823409,0.9844766,-0.9072446,-0.026288986],
[-1.1264399,-1.2674443,1.0176706,0.7478723],
[-1.4609476,-2.495958,2.6476023,-0.7192],
[1.530213,-0.017283287,-0.3597809,1.2827452],
[-1.14827,-1.4250357,2.07148,0.3654483],
[0.58386266,0.1127005,0.099308185,0.54767203],
[0.4030439,-0.1587846,-0.8817474,0.9623267],
[0.7752456,0.9753923,-0.4137564,-0.23206289],
[0.2894005,0.5173083,-1.5675887,-2.2482922],
[-0.017587805,0.8447162,0.070442095,0.27371523],
[-1.5414596,0.37476188,-0.4305615,-1.2999972],
[-0.54449975,0.25771227,-0.6583177,-0.37087512],
[0.4076065,0.48900846,-0.030393178,0.6065541],
[-0.7507286,0.25317007,-0.87461746,-0.4861526],
[-2.3279443,1.1602198,-0.3457435,-1.2076315],
[-0.7957928,1.5574993,0.3873694,-1.7760141],
[-0.21057111,0.9823828,-0.0695505,-1.1698128],
[1.2136048,-0.6689245,0.25905246,1.0968155],
[1.1964775,1.3866359,-1.3331063,-1.7890038],
[0.949396,-0.9994561,0.7066733,1.0416596],
[-0.9170172,0.26504055,-0.6306515,-0.70870286],
[1.2556156,1.493909,-1.3028557,-1.8602792],
[0.17661338,1.194818,-0.73621696,-0.22832412],
[-0.8040757,1.0260484,-0.731879,-0.3138835],
[0.8491595,1.2430257,-1.189168,-0.010135765],
[1.0702343,-1.0085405,0.45993844,0.52969736],
[-1.9785904,0.920177,-0.33150828,-2.931952],
[0.45954978,-1.1175525,1.5475287,-0.4229298],
[0.05812303,0.28565758,-1.2660983,-0.28785622],
[0.6182926,0.46280232,-1.4960349,-0.7390441],
[0.2791171,0.37509954,-1.4193124,-1.3080498],
[0.019425286,0.29683572,-1.3604033,-1.259686],
[-0.14305784,0.26853582,-1.3706715,-1.4772513],
[0.93433946,1.1689664,-1.0171713,0.10388587],
[0.22148813,0.31849968,-1.3076091,-0.36157623],
[-0.2950456,-0.2139914,-0.46951118,0.882989],
[-1.4083375,-2.4889674,2.6928744,-0.6634546],
[1.5457258,1.3852345,-1.3455665,-1.5070174],
[0.20323782,-0.15319385,-0.8902388,0.7720446],
[1.4163591,0.08125987,-0.12687661,0.9654231],
[0.05891972,0.020810166,-1.1067231,0.41193312],
[-0.077838495,1.2877551,-0.3696665,-0.32375756],
[0.4933831,0.71822774,-0.221581,0.81295013],
[0.5293924,-0.22726838,0.72446376,1.0887629],
[-1.2140415,-0.3160177,-0.18955138,0.5408541],
[-1.5048888,0.28915286,-0.43649164,-1.2832688],
[-0.74900883,-0.30728254,-0.5770657,0.31276578],
[-0.6104115,-0.2629084,-0.17195481,1.4184899],
[0.09556081,-0.2765349,-0.75993747,0.74043316],
[0.9959695,0.8649786,-0.58911747,0.6796932],
[-1.3424608,0.26120755,-0.51399636,-1.1214968],
[0.49861252,0.7322088,-0.11304578,-0.41056323],
[1.3030665,0.9198393,-0.78979576,0.1547134],
[0.8860111,-1.0812151,1.0151093,0.84146035],
[-0.2791643,1.0941811,-0.79118025,-1.0303774],
[1.2169038,-0.012741126,0.021593506,1.0311241],
[0.7602944,0.28846058,-0.17195481,0.9561291],
[-0.73816395,-0.6584386,0.017647782,0.90840274],
[0.61158913,-0.44215363,0.82945246,0.24148546],
[0.5710524,-0.47324976,0.8547651,0.42680657],
[-0.70397955,-0.12594083,-0.512215,0.16524895],
[-1.3144183,-0.2884138,-0.22019653,0.42556992],
[-0.034034174,0.8489038,-0.18698089,0.30036566],
[1.3390056,0.99389863,-0.9869206,0.051217742],
[-0.22420622,1.2514179,-1.2026204,-2.6970353],
[-0.85103524,1.6623238,0.26459035,-1.9303085],
[-1.3022398,-0.0012083998,-0.30620188,-0.57976466],
[0.29410344,1.0512077,-0.39319476,-0.48183873],
[-0.79716164,-1.3694826,0.95599264,0.66109544],
[0.8032528,0.88488656,-0.48354733,0.76770675],
[-1.1932291,-2.5490794,2.7608979,-0.6919265],
[-0.46610045,0.036885053,-0.8936077,-0.043640543],
[0.7928642,1.5208074,-1.1093074,-1.4884194],
[0.7478701,0.8387725,-0.49086425,0.75221014],
[-0.9976697,0.32653713,-0.44064504,-0.05356254],
[0.5872671,0.9785329,-0.45804316,-0.3175743],
[0.17670813,1.1535839,-0.7251412,-0.057876434],
[-2.4861255,0.8415586,-0.18955138,-3.0701413],
[-1.0069704,0.3027794,-0.6747236,-0.8202416],
[-0.22692272,-0.4208392,-0.32557702,1.1408174],
[0.05891972,-0.1367728,-0.8583961,0.6139918],
[-0.54888684,0.015913395,1.3905534,1.0571754],
[-0.680921,0.019070964,1.3011861,0.9945565],
[-1.0546669,-0.46696165,0.27684286,0.4875513],
[-1.3410568,-1.6926693,2.0360837,1.0379401],
[-0.9827536,-0.37926057,0.042568177,0.3611104],
[1.2635826,-0.26640198,-0.14229955,1.134011],
[-0.97173315,1.7140268,0.36364886,-2.0418952],
[-0.50687593,-0.3376803,-0.5568986,0.5687455],
[-2.4602592,0.8429601,-0.15060404,-3.044114],
[-2.441377,1.3157005,-0.4064418,-2.7708035],
[-0.12989233,0.18922509,-1.166232,-0.714263],
[-0.27021462,-0.41524842,1.4210117,0.91398203],
[1.4867983,-0.08436897,-0.33506703,1.4823359],
[0.04842577,-0.30972838,-0.061243698,0.29788753],
[0.83382213,1.6032755,-1.1183988,-1.5510194],
[0.1507049,0.14974704,0.6422265,0.82968336],
[0.50068325,0.23604831,-0.38469186,0.44168478],
[1.3288276,0.84540844,-0.64329624,0.9319567],
[1.1032604,1.3450643,-1.2660983,-1.092977],
[0.030491317,-0.856555,0.36503333,0.09521969],
[0.5079483,0.11165359,0.22741742,0.7478723],
[-2.314923,1.1651167,-0.39774045,-1.1747978],
[-0.7314253,-1.7405393,1.7547371,1.1606615],
[-0.7021896,-1.5288132,1.1643778,0.50614434],
[-1.9484423,0.8171086,-0.35938543,-2.558175],
[-0.3658394,-0.98128736,0.078356616,0.85571545],
[0.74323726,-1.1416649,1.1226593,0.3679264],
[-0.3025423,-0.3303436,-0.3376373,0.49250755],
[1.564608,-0.8726131,0.4413405,1.0900091],
[-0.9331266,1.8118609,0.3314139,-1.1226952],
[-1.6881292,-1.2370504,2.176261,0.37164593],
[-1.3823307,-2.427471,3.150371,0.2067776],
[-2.0615945,0.91634405,-0.3226124,-3.0583978],
[-0.40407038,-1.2706019,0.5358533,1.0577985],
[0.37054425,0.30696696,0.415451,0.80179626],
[-0.019742746,-0.44459862,-0.16661769,0.92451763],
[1.3537813,1.244782,-1.410821,-1.5063943],
[-0.7582744,-1.4606638,1.717172,0.8935245],
[-1.1023635,-1.4582324,1.3905534,0.8501409],
[-0.11040132,-0.23949848,-0.5744952,0.77266294],
[-0.87690157,-0.1280346,-0.4546812,0.04129574],
[0.69736576,0.7863602,-0.522497,0.676592],
[1.1826494,-0.9589312,0.5852788,1.0243082],
[0.12123412,0.8160616,-0.6114766,0.06051663],
[-0.75978357,-1.8729714,1.4332641,-0.25559768],
[1.2355051,-0.7031512,0.20113567,0.7986951],
[-1.0138142,-0.34676465,-0.023863131,0.34747362],
[-0.42369998,-0.62243897,0.20646583,1.1098531],
[-1.4439958,0.2388344,-0.5078655,-1.0625879],
[-0.19464067,1.1815292,-0.12964554,-1.4586535],
[0.84003437,1.2818115,-1.1100919,0.0635843],
[1.1110169,0.8985131,-0.64862645,0.66915286],
[0.6036573,-1.3467718,1.5172783,0.40263423],
[1.9320012,0.7356872,-0.9455944,0.3920987],
[0.9557134,0.18258913,-0.21011299,0.9499266],
[-0.7960737,-1.446345,0.81936884,0.3034669],
[-0.6080951,1.4282248,-1.0292392,-2.243355],
[0.68757373,0.4638492,-1.4989884,-0.82211095],
[1.2595816,-0.6755604,0.22088735,1.1141671],
[1.2205188,0.7185654,-0.91395926,0.240244],
[0.7206701,-0.47359928,0.88441575,0.49313068],
[-2.2130723,1.2437351,-0.41118592,-2.3288665],
[-0.70906866,-1.4211857,0.9909735,0.718739],
[0.3268137,-1.1119635,1.5946006,-0.33554897],
[-0.06397175,1.2692318,-1.1986747,-2.4305313],
[1.507716,0.050156962,-0.4131634,1.3162501],
[-1.060142,0.34715423,-0.42482752,-0.0460851],
[-0.15228768,-0.28212565,-0.39319476,0.57122475],
[1.0524051,0.7521167,-0.79810256,0.3939585],
[-0.3246568,-0.15459196,-0.7235722,0.29479113],
[1.1202121,-0.058855146,-0.0363233,1.4916348],
[-1.5832951,0.18817818,-0.14704366,0.2538808],
[-0.28174043,-1.2433487,0.7539298,1.3088206],
[-0.9459019,-0.322307,0.30116332,0.8005548],
[0.35278532,-0.19267696,-0.81550074,1.0596199],
[1.0634255,-1.0567651,0.46960667,0.641261],
[-0.24940573,1.0770591,-1.0709577,-0.64423376],
[0.82494265,1.548415,-1.1686087,-1.5435898],
]

nested_input = np.array(data)


def runBankNode(dpu_runner_tfBankNode, input, config):
    config = config

    print("inside the run BankNodes..")
    inputTensors = dpu_runner_tfBankNode.get_input_tensors()  #  get the model input tensor
    outputTensors = dpu_runner_tfBankNode.get_output_tensors() # get the model ouput tensor

    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    print(inputTensors[0])
    print(outputTensors[0])

    runSize = 1
    outputData = []

    outputData.append([np.zeros((runSize, outputTensors[0].dims[1]), dtype=np.float32, order="C")])

    shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])
    print('Coded shapeIn: ', shapeIn)

    #print('Input Tensor[0]: ', inputTensors[0])


    inputData = []
    inputData.append(input)


    testOutput = []
    testOutput.append(np.array([4],dtype=np.float32, order="C"))
    testInput = []
    
    #Sample which is a fake so output should be 0
    #testInput.append(np.array([1.6653357,1.0844043,-1.098024,-0.18494527], dtype=np.float32,order="C"))
    
    #Sample which is a orginal so output should be 1
    #testInput.append(np.array([-0.6005844,-0.7122524,0.70745796,0.7955938], dtype=np.float32,order="C"))
    
    ##Feeding inputs to dpu
    
    prepped_array = prep_data(nested_input)
    
    testInput.append(prepped_array[0])
    
    result_array = []

    print("Execute async")
    for i in range(prepped_array.shape[0]):
        dataInput = [prepped_array[i]]
        dataOutput = [np.empty([1], dtype=np.float32)]
        job_id = dpu_runner_tfBankNode.execute_async(dataInput, dataOutput) 
        dpu_runner_tfBankNode.wait(job_id)
        result_array.append(dataOutput)        
    
    print("Execcution completed..")
    print(len(result_array))
    print(postprocess(result_array))
    
   




def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_rounded(x):
    return np.round(sigmoid(x))    
    
def prep_data(arrayToChange):
    new_array = np.array([np.array(sub_array, dtype=np.float32) for sub_array in arrayToChange]) 
    return new_array
    
def postprocess(array):    
    post_array = []
    for element in array:
        post_array.append(sigmoid_rounded(array[element]))
    return array


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."

    root_subgraph = graph.get_root_subgraph() # Retrieves the root subgraph of the input 'graph'
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."

    if root_subgraph.is_leaf:
        return [] # If it is a leaf, it means there are no child subgraphs, so the function returns an empty list

    child_subgraphs = root_subgraph.toposort_child_subgraph() # Retrieves a listof child subgraphs of the 'root_subgraph' in topological order
    assert child_subgraphs is not None and len(child_subgraphs) > 0

    return [
        # List comprehension that filters the child_subgraphs list to include only those subgraphs that represent DPUs
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def main(argv):

    g = xir.Graph.deserialize(argv[1]) # Deserialize the DPU graph
    subgraphs = get_child_subgraph_dpu(g) # Extract DPU subgraphs from the graph
    assert len(subgraphs) == 1  # only one DPU kernel

    #Creates DPU runner, associated with the DPU subgraph.
    dpu_runners = vart.Runner.create_runner(subgraphs[0], "run")
    print("DPU Runner Created")

    # Get config
    params_path = "param.py"
    config = importlib.import_module(params_path[:-3]).parameters

    # Preprocessing

    validationset_features = np.loadtxt(r"../dataset/df_validationset_features",delimiter=',')
    validationset_labels = np.loadtxt(r"../dataset/df_validationset_labels", delimiter=',')

    print(validationset_features[0])

    input = validationset_features[0]


    # Measure time
    time_start = time.time()

    """Assigns the runBankNode function with corresponding arguments"""
    print("runBankNode -- main function intialize")
    runBankNode(dpu_runners, input, config)

    del dpu_runners
    print("DPU runnerr deleted")

    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = 1
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage : python3 dpu_infer.py <xmodel_file>")
    else:
        main(sys.argv)
