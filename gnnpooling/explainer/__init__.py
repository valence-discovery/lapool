from .backprop import *

available_explainer = dict(
    vanilla = VanillaGradExplainer,
    saliency = SaliencyExplainer,
    gradxinput = GradxInputExplainer,
    integrated = IntegrateGradExplainer,
    deconv = DeconvExplainer,
    guidedbackprop = GuidedBackpropExplainer,
    smoothgrad = SmoothGradExplainer,
    gradcam = GradCAMExplainer

)
    
def get_explainer_cls(name):
    return available_explainer[name]