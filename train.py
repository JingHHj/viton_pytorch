import torch
from dataset import GeneratorDataset
from network import Generator, VGGloss
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import pdb
from tqdm import tqdm


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print("Training on CUDA")
        else:
            print("Training on CPU")
        self.dataset = GeneratorDataset(root_dir=self.config["data_root"], image_size=self.config["image_size"])
        self.dataloader = data_loader = DataLoader(
                            self.dataset, 
                            batch_size=self.config["batch_size"], 
                            shuffle=True,  # Shuffle during training for randomness
                            num_workers=self.config["num_workers"],  # Number of CPU workers for loading data
                            pin_memory=True  # Speeds up transfer to GPU if using CUDA
                        )
        

        self.model = Generator().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.config["learning_rate"],
                                    betas=(self.config["betas"]))
        self.perpetual_loss_fn = VGGloss(self.config["layer_idx"],self.config["lambda_i"]).to(self.device)
        self.L1_loss = torch.nn.L1Loss().to(self.device)
    
    def train(self):
        # set the model to training mode
        self.model.train()
        # Initialize WandB project
        wandb.init(project="viton", name="experiment_1")

        # Log hyperparameters (optional)
        wandb.config = {
            "batch_size": self.config["batch_size"],
            "learning_rate": self.config["learning_rate"],
        }

        
        if self.config["load_checkpoint"]:
            total_loss,epoch = self.load_checkpoint()
        
        for epoch in range(self.config["num_epochs"]):
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config["num_epochs"]}")  # Add loading bar
            for batch_idx, batch in enumerate(self.dataloader):

                cloth = batch["cloth"].to(self.device)
                agostic_rep = batch["agonistic_rep"].to(self.device)
                cloth_mask = batch["cloth_mask"].to(self.device)
                image = batch["image"].to(self.device)


                # forward pass
                output = self.model(torch.concatenate([cloth, agostic_rep], dim=1))
                coars_result = output[:, :3, :, :]
                cloth_mask_result = output[:, 3, :, :].unsqueeze(1) # make sure its [B,1,H,W]
                
                # calculate loss
                mask_loss = self.L1_loss(cloth_mask_result, cloth_mask)
                coars_result_loss = self.perpetual_loss_fn(coars_result, image)

                total_loss = mask_loss + coars_result_loss

                # backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

            # Log loss to WandB
            wandb.log({
                "Total Loss": total_loss.item(),
                "Perceptual Loss": coars_result_loss.item(),
                "L1 Loss": mask_loss.item(),
                "Epoch": epoch,
                "Batch": batch_idx
            })

        print(f"Epoch [{epoch+1}/{self.config["num_epochs"]}], Loss: {total_loss.item():.4f}")


        self.save_checkpoint(epoch,total_loss)
        
        

    def save_checkpoint(self,epoch,loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, self.config["checkpoint_dir"])
        print("Checkpoint saved successful")


    def load_checkpoint(self):
        checkpoint = torch.load(self.config.model_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return loss, epoch
        

    def save_inference(self):
        pass
if __name__ == "__main__":
    pass
