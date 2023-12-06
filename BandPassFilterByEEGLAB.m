function data = BandPassFilterByEEGLAB(data,b)
    %data is the input forom gtec (nTimepts, nchan)
    %b is the filtering coefficients obtained using EEGlab
    EEG = struct;
    EEG.data = data'; 
    EEG.event = [];
    EEG.trials = 1;
    EEG.pnts   = size(data,1);
    EEG = firfilt(EEG,b);
    data = EEG.data';
end

