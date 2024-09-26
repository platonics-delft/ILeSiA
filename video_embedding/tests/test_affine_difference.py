

from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from video_embedding.utils import all_trial_names, get_session, set_session
from video_embedding.models.video_embedder import RiskyBehavioralVideoEmbedder
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, StampedDistLatentObservationsRiskLabels, StampedLatentObservationsRiskLabels, VideoObservationsRiskAndSafeLabels, LatentObservationsRiskLabelsPriorRisk, StampedDistLatentObservationsRiskLabelsPriorRisk, StampedLatentObservationsRiskLabelsPriorRisk
from scipy.spatial.distance import cosine


skill_name = 'peg_door404'
session = 'manipulation_demo404_session'
framedrop_policy = OnlyLabelledFramesDroppingPolicy

video_latent_dim = 16
set_session(session)

# load dataset not shuffled

video_embedder = RiskyBehavioralVideoEmbedder(name=skill_name, latent_dim=video_latent_dim)
video_embedder.load_model()
dataloader = video_embedder.load_dataset(train_names=[skill_name],AUGMENT_VIDEOS=False)

# video_names = all_trial_names(skill_name)

# train_dataloader, test_dataloader = RiskEstimationDataset.load(
#     video_names=video_names,
#     video_embedder=video_embedder,
#     batch_size=video_embedder.batch_size,
#     frame_dropping_policy=framedrop_policy,
#     features=LatentObservationsRiskLabels,
# )
# train_dataloader_images, test_dataloader_images = RiskEstimationDataset.load(
#     video_names=video_names,
#     video_embedder=video_embedder,
#     batch_size=video_embedder.batch_size,
#     frame_dropping_policy=framedrop_policy,
#     features=VideoObservationsRiskAndSafeLabels,
# )
# X_test_images, Y_test_images = RiskEstimationDataset.dataloader_to_array(test_dataloader_images)
# X_train_images, Y_train_images = RiskEstimationDataset.dataloader_to_array(train_dataloader_images)

# load dataset not shuffled but with translation 
dataloader_augment = video_embedder.load_dataset(train_names=[skill_name],AUGMENT_VIDEOS=True)

# go though frames and get difference 

numbers = []
for b1, b2 in zip(dataloader, dataloader_augment):
    b1 = b1[0]
    for f1, f2 in zip(b1, b2):
        
        l1 = video_embedder.model.encoder(f1.unsqueeze(0))
        l2 = video_embedder.model.encoder(f2.unsqueeze(0))

        numbers.append( cosine(l1.detach().cpu().numpy().squeeze(), l2.detach().cpu().numpy().squeeze()) )
        
print(sum(numbers)/len(numbers))
