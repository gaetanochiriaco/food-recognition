from timm.data import  Mixup
import torch
import time
from timm.loss import SoftTargetCrossEntropy
import torch.nn as nn
from food_recognition.bert_emb import get_bert_embeddings

def training_loop(model,
                  loader,
                  epochs,
                  optimizer,
                  scheduler = None,
                  last_batch=0,
                  use_scaler=False,
                  mixup=False,
                  test=False,
                  test_loader= None,
                  bert=False,
                  label_path= None,
                  print_batch=100,
                  save = False,
                  save_path = "",
                  model_name = "model",
                  num_classes = 2000):
  

  tot_batch = last_batch
  # total number of epochs
  last_epoch = tot_batch //len(loader)
  batch_size = loader.batch_size
  # use grad scaler
  if use_scaler:
    scaler = torch.cuda.amp.GradScaler()
  if mixup:
    train_loss_fn = SoftTargetCrossEntropy().cuda()

    mixup_args = dict(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode="batch",
            label_smoothing=0.1, num_classes=num_classes)
    mixup_fn = Mixup(**mixup_args)


  else:
    train_loss_fn = nn.CrossEntropyLoss().cuda()
  validate_loss_fn = nn.CrossEntropyLoss().cuda()
  # use bert loss
  if bert:

    bert_emb = get_bert_embeddings(label_path)
    bert_emb = bert_emb.cuda()
    MSE_loss_fn = nn.MSELoss().cuda()

    activation = {}
    def get_activation(name):
      def hook(model, input, output):
        activation[name] = output.detach()
      return hook

    model.layers[3].blocks[1].mlp.fc2.register_forward_hook(get_activation('layers.3.blocks.1.mlp.fc2'))



  # Training loop
  for i in range(last_epoch+1,epochs+1):
    start_epoch = time.time()

    # Compute accuracy if not mixup
    if not mixup:
      trn_corr = 0

    # Training batches
    for b, (X_train, y_train) in enumerate(loader):
      b+=1
      tot_batch+=1

      X_train, y_train = X_train.cuda(), y_train.cuda()

      # Apply mixup
      if mixup:
        X_train, y_train = mixup_fn(X_train, y_train)

      X_train = X_train.contiguous(memory_format=torch.channels_last)

      # Use grad scaler (float16 precision)
      if use_scaler:
        with torch.cuda.amp.autocast():
          y_pred = model(X_train)
          loss = train_loss_fn(y_pred, y_train)
          if bert:
            img_emb = nn.functional.tanh(activation['layers.3.blocks.1.mlp.fc2'].mean(dim=1))
            if mixup:
              label_emb = (y_train@bert_emb)
    
            else:
              label_emb = bert_emb[y_train,:]
          
            mse_loss = MSE_loss_fn(img_emb, label_emb)
            loss = 0.6*loss + 0.4* mse_loss


      else:
        y_pred = model(X_train)
        loss = train_loss_fn(y_pred, y_train)
        if bert:
          img_emb = nn.functional.tanh(activation['layers.3.blocks.1.mlp.fc2'].mean(dim=1))
          if mixup:
            label_emb = (y_train@bert_emb)
    
          else:
            label_emb = bert_emb[y_train,:]
          
          mse_loss = MSE_loss_fn(img_emb, label_emb)
          loss = 0.6*loss + 0.4* mse_loss

      # Calculate accuracy
      if not mixup:
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train.cuda()).sum()
        trn_corr += batch_corr

      # Backprop
      optimizer.zero_grad()
      if use_scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      
      else:
        loss.backward()
        optimizer.step()
      
      # Scheduler steps
      scheduler.step(tot_batch//len(loader))

      # Print every print_batch
      if b%print_batch== 0:
        if mixup:
          print(f'epoch: {i:2}  batch: {b:4} [{batch_size*b:6}/{(len(loader)*batch_size)}]  loss: {loss.item():10.8f}')
    
        else: 
          print(f'epoch: {i:2}  batch: {b:4} [{batch_size*b:6}/{(len(loader)*batch_size)}]  loss: {loss.item():10.8f} accuracy: {batch_corr.item()*100/(len(predicted)):7.3f}% avg acc: {trn_corr.item()*100/(b*batch_size):7.3f}%')

        # Save every print_batch
        if save:
          torch.save({
            'epoch': i,
            'batch':tot_batch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            }, save_path + f"/{model_name}.pt")


    #if test is True calculate metrics on test set
    if test:
      with torch.no_grad():
        tst_corr = 0
        for b, (X_test, y_test) in enumerate(test_loader):
          y_val = model(X_test.cuda())

          predicted = torch.max(y_val.data, 1)[1] 
          tst_corr += (predicted == y_test.cuda()).sum()
            
          loss = validate_loss_fn(y_val, y_test.cuda())
          tot_loss+=loss.detatch.cpu()

      print(f'Epoch end: {i:2}  val loss: {tot_loss/len(test_loader):10.8f} val accuracy: {tst_corr*100/(len(test_loader)*batch_size):7.3f}%  time : {time.time()-start_epoch:0f} s')
