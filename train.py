import argparse
import torch
from pathlib import Path
import os
from utils.utils import *
from torch.utils.data import DataLoader
from utils.models import *
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--content_dir', type = str, default = r"C:\Users\Hp World\OneDrive\Desktop\NST-AI FILTER\content_data" ,
                        help = "Location of content dataset")
    
    parser.add_argument('--style_dir', type = str, default = r"C:\Users\Hp World\OneDrive\Desktop\NST-AI FILTER\style_data" ,
                        help = "Location of style dataset")
    
    parser.add_argument("--vgg" , type = str , default = r"C:\Users\Hp World\OneDrive\Desktop\NST-AI FILTER\vgg_normalised.pth",
                        help = " Location of pretrained VGG" )
    
    parser.add_argument("--experiment" , type = str , default = "experiment_1" , help = " Name of experiment " )


    parser.add_argument("--final_size" , type = int , default=256 ,
                        help = "size of final image")
    
    
    parser.add_argument("--content_size" , type = int , default= 512 ,
                        help = "size of content image")
    
    
    parser.add_argument("--style_size" , type = int , default= 512 ,
                        help = "size of style image")
    
    
    parser.add_argument("--crop" , action = "store_true", default=True ,
                        help = "crop_image")
    
    parser.add_argument("--batch_size" , type=int , default = 2 , help = "Batch_size" )

    parser.add_argument("--lr" , type = float, default = 1e-4, help = "Learning rate" )

    parser.add_argument("--lr_decay" , type = float , default = 5e-5, help = "Learning rate decay" )

    parser.add_argument("--epochs" , type = int , default = 2, help = "Number of epoch" )
    
    parser.add_argument("--content_weight" , type = float , default = 1.0, help = "Content weight" )

    parser.add_argument("--style_weight" , type = float , default = 10 , help = "Style weight" )

    parser.add_argument("--log_interval" , type = float , default = 1 , help = "log intervel" )

    parser.add_argument("--save_intervel" , type = int , default = 2 , help = "save intervel" )

    
    parser.add_argument("--resume" , action = "store_true" , default= False ,
                        help = "Resume training")
    
    
    parser.add_argument("--decoder_path" , type = str , default= None ,
                        help = "decoder path")
    
    
    parser.add_argument("--optimizer_path" , type = str , default= None ,
                        help = "path optimizer")

    return parser.parse_args()
     

    

def main():
    args = parse_arguments()
    # print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(exist_ok = True, parents = True )


    # saving arguments values in a text files 

    with open(save_dir / "args.txt" , "w") as args_file:
       for key , value in vars(args).items():
           args_file.write(f"{key}:{value}\n")

    
    content_transform = get_transform(args.content_size,args.crop,args.final_size)
    style_transform = get_transform(args.style_size,args.crop,args.final_size)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset,batch_size=args.batch_size,shuffle = True,pin_memory = True,drop_last=True)

    style_dataloader = DataLoader(style_dataset,batch_size=args.batch_size,shuffle = True,pin_memory = True,drop_last=True)

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer  = optim.Adam(decoder.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lambda epoch: 1.0/(1.0 + args.lr_decay*epoch)
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))


    print("Training")

    mse_loss = torch.nn.MSELoss()
    encoder.eval()
    running_loss = None
    running_conloss = None
    running_styloss = None

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader,style_dataloader),
                             total = min(len(content_dataloader),len(style_dataloader)))
        running_loss = 0
        running_conloss = 0
        running_styloss = 0
        for content_batch,style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            print(len(c_feats))
            print(len(s_feats))
            print(c_feats[0].shape)

            t = adaptive_instance_normalization(c_feats[-1] ,s_feats[-1])

            g = decoder(t)
            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1],t)*args.content_weight
            loss_s = 0
            for g_f,s_f in zip(g_feats,s_feats):
              g_mean,g_std = calc_mean_std(g_f)
              s_mean,s_std = calc_mean_std(s_f)
              loss_+= mse_loss(g_mean,s_mean) + mse_loss(g_std , s_std)
              
            loss_s = loss_s * args.style_weight
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Loss : {loss.item() :4f} , content_loss {loss_c.item() :4f} , style_loss {loss_s.item():4f}")

            running_loss += loss.item()
            running_conloss +=loss_c.item()
            running_styloss +=loss_s.item()

        scheduler.step()

        running_loss /=len(content_dataloader)
        running_conloss /= len(content_dataloader)
        running_styloss /= len(content_dataloader)

        if(epoch+1) % args.log_interval == 0:
            tqdm.write(f"Iter {epoch+1}: Loss {running_loss :4f} , content_loss {running_conloss :4f} , style_loss {running_styloss :4f}")

        if(epoch+1) % args.save_intervel == 0:
            torch.save(decoder.state_dict() , save_dir / f"decoder_{epoch+1}.pth")
            torch.save(optimizer.state_dict(),save_dir / f"optimizer{epoch+1}.pth")


            with torch.no_grad():
                output = torch.cat([content_batch,style_batch,g],dim=0)
                save_image(output,save_dir /f"output_{epoch+1}.png",nrow = args.batch_size)









            


if __name__ == '__main__':
    main()
