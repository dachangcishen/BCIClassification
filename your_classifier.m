function state = your_classifier(data, Mdl , SavedFilter)

    %predict the class of the data using your trained model
    %data in a format of (SampleRate x ChanNum)

    %Extract feature using SavedFilter
    X = [];
    for i = 1:length(SavedFilter)
        Signal = BandPassFilterByEEGLAB(data,SavedFilter{i});
        Power  = sum(Signal.^2,1)/size(Signal,1);
        X = [X Power];
    end
    
    %Class returns prediction of trained model
    %0: Rest, 1:Left Hand, 2:Right Hand, 3:Jump
    
    Class = predict(Mdl, X);
    
    switch Class %REST:0, ROTATE = 11, JUMP = 12, SLIDE = 13
        case 0
            state = 0;
        case 1
            state = 11;
        case 2
            state = 12;
        case 3
            state = 13;
    end
 





