class TCN_trainer:
    def __init__(self,model, train_loader, test_loader,num_epochs = None, lr = None,batch_size=BATCH_SIZE, verbose = 1, patience=None):
        self.model = model
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.test_loader =test_loader
        self.verbose = verbose
        self.patience =patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer =  optim.Adam(model.parameters(), lr = lr)
        self.scheduler =   optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.nb_epochs = num_epochs
        self.train_hist  = np.zeros(self.nb_epochs)
        self.vaild_hist =[]

    def train(self):
        config = {
        "learning_rate": IR,
        "epochs": EPOCH,
        "batch_size": BATCH_SIZE,
        "shuffle":SUFFLE,
        'verbose':1,
        'patience':PATIENCE,
        'dropout':DROPOUT,
        'target':TARGET,
        'feature_columns':FEATURE,
        "NUM_LAYERS":NUM_LAYERS,
         'HIDDEN_SIZE':HIDDEN_SIZE,
         'DATA':data_list[DATA]

        }
        run = wandb.init(project ="LSTM",config=config)

        self.model.to(self.device)
        for epoch in range(self.nb_epochs):
            avg_cost = 0
            total_batch = len(self.train_loader)
            for batch_idx, samples in enumerate(self.train_loader):
                x_train, y_train = samples
                x_train =x_train.cuda()
                y_train =y_train.cuda()
                outputs = self.model(x_train)
                loss = self.criterion(outputs, y_train)
                self.optimizer.zero_grad()
                ##loss
                loss.backward()
                self.optimizer.step()
                avg_cost += loss/total_batch
                ##정확도 계산
                predict, label = outputs.clone().detach(),y_train.clone().detach()
                predict = inverse_min_max(predict.to('cpu').numpy(),min_max_list)
                label =  inverse_min_max(label.to('cpu').numpy(),min_max_list)
                score = self.MAE(predict,label)
                acc = self.accuracy(label,predict)
                wandb.log({"Training Loss": loss.item()})
            self.train_hist[epoch] = avg_cost
            if epoch % self.verbose == 0:
                total_loss, score,acc =self.vaild()
                print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))
                print('vaild_loss:', '{:.4f}'.format (total_loss), 'MAE :', '{:.4f}'.format(score),'Accuarcy :', '{:.4f}'.format(acc))
                wandb.log({"Evaluation Loss": loss.item()})
                wandb.log({"Evaluation MAE": score.item()})
                wandb.log({"Evaluation Accuracy": acc.item()})
                self.vaild_hist.append({"vaild_loss":total_loss ,"vaild_score":score,"vaild_accuracy":acc,"epoch":epoch})
                self.scheduler.step(total_loss)
        # patience번째 마다 early stopping 여부 확인
            if (epoch % self.patience == 0) & (epoch != 0):
                # loss가 커졌다면 early stop
                index =int(epoch/self.patience)
                if index> 1:
                    print((self.vaild_hist[index-1]['vaild_loss'] ,  self.vaild_hist[index]['vaild_loss'])   )
                    if self.vaild_hist[index-1]['vaild_loss']<self.vaild_hist[index]['vaild_loss']:
                        print('\n Early Stopping')
                        break
                    else:
                        print('model was saved')
                        torch.save(self.model,"best-model.pt")
    def vaild(self):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = []
            avg_cost = 0
            score = 0
            acc = 0
            total_batch= len(test_dataloader)
            for batch_idx, samples in enumerate(self.test_loader):
                x_test, y_test = samples
                x_test =x_test.cuda()
                y_test =y_test.cuda()
                outputs = self.model(x_test)
                loss = self.criterion(outputs, y_test)
                avg_cost += loss
                predict = inverse_min_max(outputs.to('cpu'),min_max_list)
                y_test =  inverse_min_max(y_test.to('cpu'),min_max_list)
                score += self.MAE(predict,y_test)
                acc += self.accuracy(y_test,predict)
        total_loss = avg_cost/total_batch
        score = score/len(self.test_loader)
        acc = acc/len(self.test_loader)
        self.model.train()
        return total_loss,score,acc

    def MAE(self,true, pred):
         return np.mean(np.abs(true-pred))
    def accuracy(self,true, pred):
        return (1-np.mean(np.abs((true-pred)/true)))


