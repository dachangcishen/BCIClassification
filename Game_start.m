clear all;
clc;
global AcquisitionTime
AcquisitionTime=inf;
global ChanNum
ChanNum=1:4;
global FileName
FileName='Test';
global SampleRate
SampleRate=256;
global ai
global EEGDaq_StartTime


[EEGDaq_StartTime, ai]=openUSBdaq(SampleRate, AcquisitionTime, FileName, ChanNum, 49,2);
pause(2);


msgbox('gUSBamp turning on ');
disp('>>>>>>>>>>>>>>>>>>>>>>>>>>>USB_OPENED');
index=1;

%initialize any preloaded files
model_dir = 'G:\BMEG3330\project_2\BME3330\BME_3330\Psychtoolbox_3330\ALL_SCRIPTS_CURRENT_VERSION\training_data\group2';
load([model_dir, '/Trained_Mdl.mat']);
load([model_dir, '/SavedFilter.mat']);
%initialize UDP port
PortAddress = '169.254.129.252';
hudps=dsp.UDPSender('RemoteIPAddress', PortAddress, 'RemoteIPPort', 5555);

while index
  %extract 1s data for each loop 
  Data_Present=peekdata(ai, SampleRate);
  %%pelase add your control algorithms here!!! For different state, please
  %your_classifier should return predictive labels correpsonding to the
  %class, by default, REST = 0, ROTATE = 11, JUMP = 12, SLIDE = 13
  
  state = your_classifier(Data_Present, Mdl, SavedFilter);
  disp(state)
  step(hudps, uint8(state));
  
  
end
release(hudps);