from timm.data import  Mixup
import torch
import time
from timm.loss import SoftTargetCrossEntropy
import torch.nn as nn
from food_recognition.bert_emb import get_bert_embeddings
import ttach as tta
import numpy as np

def training_loop_time_test(model,
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
                  bert_model = 'bert-large-uncased',
                  label_path= None,
                  print_batch=100,
                  save = False,
                  save_path = "",
                  model_name = "model",
                  num_classes = 2000,
                  deit = False,
                  teacher = None):
  

  tot_batch = last_batch
  # total number of epochs
  last_epoch = tot_batch //len(loader)
  batch_size = loader.batch_size
  if deit:
    teacher = teacher.cuda()
    teacher.eval()
  # use grad scaler
  if use_scaler:
    scaler = torch.cuda.amp.GradScaler()
  if mixup:
    train_loss_fn = SoftTargetCrossEntropy().cuda()

    mixup_args = dict(
            mixup_alpha=0.5, cutmix_alpha=0.5, cutmix_minmax=None,
            prob=0.5, switch_prob=0, mode="batch",
            label_smoothing=0.1, num_classes=num_classes)
    mixup_fn = Mixup(**mixup_args)


  else:
    train_loss_fn = nn.CrossEntropyLoss().cuda()
  validate_loss_fn = nn.CrossEntropyLoss().cuda()
  deit_loss_fn = nn.CrossEntropyLoss().cuda()
  # use bert loss
  if bert:

    bert_emb = get_bert_embeddings(label_path,model_name  = bert_model)
    bert_emb = bert_emb.cuda()
    MSE_loss_fn = nn.MSELoss().cuda()

    activation = {}
    def get_activation(name):
      def hook(model, input, output):
        activation[name] = output.detach()
      return hook

    try:
      model.layers[3].blocks[1].mlp.fc2.register_forward_hook(get_activation('emb'))
    except:
      model.blocks[-1].mlp.fc2.register_forward_hook(get_activation('emb'))
      


  # Training loop
  for i in range(last_epoch+1,epochs+1):
    tot_time = 0
    start_epoch = time.time()

    # Compute accuracy if not mixup
    if not mixup:
      trn_corr = 0
      
    start_time_b = time.time()
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
          if deit:
            y_pred,distil = model(X_train)
            with torch.no_grad():
              teach_pred = teacher(X_train)
              teach_labels = torch.max(teach_pred.data, 1)[1]
            loss = 0.5*train_loss_fn(y_pred, y_train) + 0.5*deit_loss_fn(distil,teach_labels)
          else:  
            y_pred = model(X_train)

            loss = train_loss_fn(y_pred, y_train)
          if bert:
            img_emb = nn.functional.tanh(activation['emb'].mean(dim=1))
            if mixup:
              label_emb = (y_train@bert_emb)
    
            else:
              label_emb = bert_emb[y_train,:]
          
            mse_loss = MSE_loss_fn(img_emb, label_emb)
            loss = 0.6*loss + 0.4* mse_loss


      else:
        if deit:
          y_pred,distil = model(X_train)
          with torch.no_grad():
            teach_pred = teacher(X_train)
            teach_labels = torch.max(teach_pred.data, 1)[1]
          loss = 0.5*train_loss_fn(y_pred, y_train) + 0.5*deit_loss_fn(distil,teach_labels)
        else:  
          y_pred = model(X_train)

          loss = train_loss_fn(y_pred, y_train)
        if bert:
          img_emb = nn.functional.tanh(activation['emb'].mean(dim=1))
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
      
      end_time_b = time.time()
      print((end_time_b - start_time_b)/b)
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
                  bert_model = 'bert-large-uncased',
                  label_path= None,
                  print_batch=100,
                  save = False,
                  save_path = "",
                  model_name = "model",
                  num_classes = 2000,
                  deit = False,
                  teacher = None):
  

  tot_batch = last_batch
  # total number of epochs
  last_epoch = tot_batch //len(loader)
  batch_size = loader.batch_size
  if deit:
    teacher = teacher.cuda()
    teacher.eval()
  # use grad scaler
  if use_scaler:
    scaler = torch.cuda.amp.GradScaler()
  if mixup:
    train_loss_fn = SoftTargetCrossEntropy().cuda()

    mixup_args = dict(
            mixup_alpha=0.5, cutmix_alpha=0.5, cutmix_minmax=None,
            prob=0.5, switch_prob=0, mode="batch",
            label_smoothing=0.1, num_classes=num_classes)
    mixup_fn = Mixup(**mixup_args)


  else:
    train_loss_fn = nn.CrossEntropyLoss().cuda()
  validate_loss_fn = nn.CrossEntropyLoss().cuda()
  deit_loss_fn = nn.CrossEntropyLoss().cuda()
  # use bert loss
  if bert:

    bert_emb = get_bert_embeddings(label_path,model_name  = bert_model)
    bert_emb = bert_emb.cuda()
    MSE_loss_fn = nn.MSELoss().cuda()

    activation = {}
    def get_activation(name):
      def hook(model, input, output):
        activation[name] = output.detach()
      return hook

    try:
      model.layers[3].blocks[1].mlp.fc2.register_forward_hook(get_activation('emb'))
    except:
      model.blocks[-1].mlp.fc2.register_forward_hook(get_activation('emb'))
      


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
          if deit:
            y_pred,distil = model(X_train)
            with torch.no_grad():
              teach_pred = teacher(X_train)
              teach_labels = torch.max(teach_pred.data, 1)[1]
            loss = 0.5*train_loss_fn(y_pred, y_train) + 0.5*deit_loss_fn(distil,teach_labels)
          else:  
            y_pred = model(X_train)

            loss = train_loss_fn(y_pred, y_train)
          if bert:
            img_emb = nn.functional.tanh(activation['emb'].mean(dim=1))
            if mixup:
              label_emb = (y_train@bert_emb)
    
            else:
              label_emb = bert_emb[y_train,:]
          
            mse_loss = MSE_loss_fn(img_emb, label_emb)
            loss = 0.6*loss + 0.4* mse_loss


      else:
        if deit:
          y_pred,distil = model(X_train)
          with torch.no_grad():
            teach_pred = teacher(X_train)
            teach_labels = torch.max(teach_pred.data, 1)[1]
          loss = 0.5*train_loss_fn(y_pred, y_train) + 0.5*deit_loss_fn(distil,teach_labels)
        else:  
          y_pred = model(X_train)

          loss = train_loss_fn(y_pred, y_train)
        if bert:
          img_emb = nn.functional.tanh(activation['emb'].mean(dim=1))
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


def testing_loop(model,
                 loader,
                 TTA=True,
                 print_batch=100,
                 get_dict = False):
  
  batch_size = loader.batch_size
  if TTA:
    transforms = tta.Compose([tta.HorizontalFlip(),
                            tta.VerticalFlip(),
                            tta.Rotate90(angles=[0, 90,180])])

    tta_model = tta.ClassificationTTAWrapper(model, transforms)

  dict_error = {}
  model.cuda().eval()
  tst_corr = 0
  tst_5_corr = 0
  with torch.no_grad():
    for b,batch in enumerate(loader):
        b+=1
        image,label=batch
        image = image.cuda()
        image = image.contiguous(memory_format=torch.channels_last)
        if TTA:
          pred = tta_model(image)
        
        else:
          pred = model(image)

        pred_1 = torch.argmax(pred,dim=1).detach().cpu().numpy()
        _, pred_5 = torch.topk(pred, dim=1,k=5)
        pred_5 = pred_5.cpu().numpy()
  
        label = label.cpu().numpy()
        corr_pred = (label == pred_1)
        num_corr = corr_pred.sum()
        if get_dict:
          lab_error = [label[i] for i in range(len(label)) if corr_pred[i]==False]
          for i in lab_error:
            try:
              dict_error[str(i)] = dict_error[str(i)] + 1
            except:
              dict_error[str(i)] = 1
        
        tst_corr += num_corr

        num_top5_corr =  (label[...,None] == pred_5).any(axis=1).sum()
        tst_5_corr += num_top5_corr
        
        if b%print_batch == 0:
          print("Top1 Accuracy:",(tst_corr*100)/(batch_size*b),"\tTop5 Accuracy",(tst_5_corr*100)/(batch_size*b))
          
  if get_dict:
      return dict_error,(tst_corr*100)/(batch_size*b)
  else:
      return (tst_corr*100)/(batch_size*b)

