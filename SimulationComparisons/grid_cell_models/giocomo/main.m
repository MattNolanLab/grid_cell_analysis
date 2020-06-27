% Giocomo, Zilli, Fransen, and Hasselmo 2007's temporal interference model
% Adapted from Zilli 2012
% This code is released into the public domain. Not for use in skynet.


% if >0, plots the sheet of activity during the simulation on every livePlot'th step

basePhases = 0:6
spikeCoords = cell(length(basePhases), 1);
actualHdAtSpike = cell(length(basePhases), 1);
spikeTimes = cell(length(basePhases), 1);

for basePhase=basePhases
	watchCell = basePhase + 1;

	livePlot = 000;

	% if =0, just give constant velocity. if =1, load trajectory from disk
	useRealTrajectory = 1;
	constantVelocity = 1*[.5; 0*0.5]; % m/s

	%% Simulation parameters
	dt = .02; % time step, s
	simdur = 1293; % total simulation time, s  %
	tind = 1; % time step number for indexing
	t = 0; % simulation time variable, s
	x = 0; % position, m
	y = 0; % position, m

	%% Model parameters
	ncells = 1;
	% Basline maintains a fixed frequency
	baseFreq = 6.42; % dorsal, Hz
	% baseFreq = 4.23; % ventral, Hz
	% Directional preference of each dendrite (this also sets the number of dendrites)
	dirPreferences = [0 2*pi/3 4*pi/3];
	% Scaling factor relating speed to oscillator frequencies
	beta = 0.385; % Hz/(m/s) 
	spikeThreshold = 1.8;


	%% History variables
	speed = zeros(1,ceil(simdur/dt));
	curDir = zeros(1,ceil(simdur/dt));
	vhist = zeros(1,ceil(simdur/dt));
	fhist = zeros(1,ceil(simdur/dt));

	%% Firing field plot variables
	nSpatialBins = 60;
	minx = 0; maxx = 1.1; % m
	miny = 0; maxy = 1.1; % m
	occupancy = zeros(nSpatialBins);
	spikes = zeros(nSpatialBins);

	%spikeTimes = [];
	%spikeCoords = [];
	spikePhases = [];
	%actualHdAtSpike = [];

	%% Initial conditions
	% Oscillators will start at phase 0:
	dendritePhases = zeros(1,length(dirPreferences)); % rad
	%basePhase = 0; % rad
	disp(basePhase)

	%% Make optional figure of sheet of activity
	if livePlot
	  h = figure('color','w','name','Activity of one cell');
	  if useRealTrajectory
	    set(h,'position',[520 378 1044 420])
	  end
	  drawnow
	end

	%% Possibly load trajectory from disk
	if useRealTrajectory
	  load ../trajectory_data.mat;
	  % interpolate down to simulation time step
	  %pos = [interp1(pos(3,:),pos(1,:),0:dt:pos(3,end));
	%	 interp1(pos(3,:),pos(2,:),0:dt:pos(3,end));
	%	 interp1(pos(3,:),pos(3,:),0:dt:pos(3,end));
	%	 interp1(pos(3,:),pos(4,:),0:dt:pos(3,end))];  % hd reference
	  pos(1:2,:) = pos(1:2,:)/100; % cm to m
	  vels = [diff(pos(1,:)); diff(pos(2,:))]/dt; % m/s
	  x = pos(1,1); % m
	  y = pos(2,1); % m
	  actualHd = pos(4, :); % real hd for reference
	end

	%% !! Main simulation loop
	fprintf('Simulation starting. Press ctrl+c to end...\n')
	while t<simdur
	  tind = tind+1;
	  t = dt*tind;
	  
	  % Velocity input
	  if ~useRealTrajectory
	    v = constantVelocity; % m/s
	  else
	    v = vels(:,tind); % m/s
	  end
	  curDir(tind) = atan2(v(2),v(1)); % rad
	  speed(tind) = sqrt(v(1)^2+v(2)^2);%/dt; % m/s
	  
	  x(tind) = x(tind-1)+v(1)*dt; % m
	  y(tind) = y(tind-1)+v(2)*dt; % m
	    
	  % Dendrite frequencies are pushed up or down from the basline frequency
	  % depending on the speed and head direction, with a scaling factor
	  % baseFreq*beta that sets the spacing between the spatial grid fields.
	  dendriteFreqs = baseFreq + baseFreq*beta*speed(tind)*cos(curDir(tind)-dirPreferences); % Hz

	  % Advance oscillator phases
	  % Radial frequency (2pi times frequency in Hz) is the time derivative of phase.
	  dendritePhases = dendritePhases + dt*2*pi*dendriteFreqs; % rad
	  basePhase = basePhase + dt*2*pi*baseFreq; % rad
	  
	  % Sum each dendritic oscillation separately with the baseline oscillation
	  dendritePlusBaseline = cos(dendritePhases) + cos(basePhase);
	    
	  % Final activity is the product of the oscillations.
	  % Note this rule has some odd features such as positive
	  % activity given an even number of negative oscillator sums and
	  % the baseline is included separately in each term in the product.
	  f = prod(dendritePlusBaseline);
	  
	  % threshold threshold f
	  f = f.*(f>0);
	  
	  % Save for later
	  fhist(tind) = f;
	  
	  % Save firing field information
	  if f>spikeThreshold
	    spikeTimes{watchCell} = [spikeTimes{watchCell}; t];
	    spikeCoords{watchCell} = [spikeCoords{watchCell}; x(tind) y(tind)];
	    spikePhases = [spikePhases; basePhase];
	    actualHdAtSpike{watchCell} = [actualHdAtSpike{watchCell}; actualHd(tind)];
	  end
	  if useRealTrajectory
	    xindex = round((x(tind)-minx)/(maxx-minx)*nSpatialBins)+1;
	    yindex = round((y(tind)-miny)/(maxy-miny)*nSpatialBins)+1;
	    occupancy(yindex,xindex) = occupancy(yindex,xindex) + dt;
	    spikes(yindex,xindex) = spikes(yindex,xindex) + double(f>spikeThreshold);
	  end
	  
	  if livePlot>0 && (livePlot==1 || mod(tind,livePlot)==1)
	    if ~useRealTrajectory
	      figure(h);
	      subplot(121);
	      plot(fhist(1:tind));
	      title('Activity');
	      xlabel('Time (s)')
	      axis square
	      set(gca,'ydir','normal')
	      title(sprintf('t = %.1f s',t))
	      subplot(122);
	      plot(x(1:tind),y(1:tind))
	      hold on;
	      if ~isempty(spikeCoords)
		cmap = jet;
		cmap = [cmap((end/2+1):end,:); cmap(1:end/2,:)];
		phaseInds = mod(spikePhases,2*pi)*(length(cmap)-1)/2/pi;
		pointColors = cmap(ceil(phaseInds)+1,:);
	  
		scatter3(spikeCoords(:,1), ...
			 spikeCoords(:,2), ...
			 zeros(size(spikeCoords(:,1))), ...
			 30*ones(size(spikeCoords(:,1))), ...
			 pointColors, ...
			 'o','filled');
	      end
	      axis square
	      title({'Trajectory (blue) and',...
		     'spikes (colored by theta phase',...
		     'blues before baseline peak, reds after)'})
	      drawnow
	    else
	      figure(h);
	      subplot(131);
	      plot((0:tind-1)*dt,fhist(1:tind));
	      hold on;
	      plot([0 tind-1]*dt,[spikeThreshold spikeThreshold],'r')
	      title('Activity (blue) and threshold (red)');
	      xlabel('Time (s)')
	      axis square
	      set(gca,'ydir','normal')
	      subplot(132);
	      imagesc(spikes./occupancy);
	      axis square
	      set(gca,'ydir','normal')
	      title({'Rate map',sprintf('t = %.1f s',t)})
	      subplot(133);
	      plot(x(1:tind),y(1:tind))
	      hold on;
	      if ~isempty(spikeCoords)
		cmap = jet;
		cmap = [cmap((end/2+1):end,:); cmap(1:end/2,:)];
		phaseInds = mod(spikePhases,2*pi)*(length(cmap)-1)/2/pi;
		pointColors = cmap(ceil(phaseInds)+1,:);
	  
		scatter3(spikeCoords(:,1), ...
			 spikeCoords(:,2), ...
			 zeros(size(spikeCoords(:,1))), ...
			 30*ones(size(spikeCoords(:,1))), ...
			 pointColors, ...
			 'o','filled');
	      end
	      axis square
	      title({'Trajectory (blue) and',...
		     'spikes (colored by theta phase',...
		     'blues before baseline peak, reds after)'})
	      drawnow
	    end
	  end  
	end
end

save('results.mat')
disp('done');
