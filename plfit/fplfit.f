c FILE: fplfit.f
      subroutine plfit(x,nosmall,ksa,av,lx)
c     internal section of plfit
c      requires that x be sorted!
      integer lx,nosmall
      real*8 x(lx)
      real*8 ksa(lx)
      real*8 av(lx)
cf2py intent(in) :: x, nosmall
cf2py intent(hide) :: lx
cf2py intent(out) :: av,ksa
      integer j,n,k,nk
      real*8 lzs,cx,cf,ks,ksp,xmin
      xmin = 0.0
c     write(*,*) "Starting length ",lx," run"
c     equivalent to python loop: i in arange(1,lx)      
      do 100 i=1,lx
            if ( x(i) .lt. xmin ) write(*,*) "WARNING X WAS NOT SORTED!"
c           skip repeat values (only compute unique values of xmin)            
            if ( x(i) .eq. xmin ) then
                ksa(i)=0
                av(i)=0
                goto 100
            endif
            xmin = x(i)
            lzs = 0.0
            do 200 j=i,lx
                lzs = lzs + log(x(j)/xmin)
200         continue            
            n = j - i
            av(i) = float(n) / (lzs)
            if (nosmall.gt.0) then
                if ((av(i)-1.0)/sqrt(float(n+1)) .ge. 0.1) then
c                   write(*,*) "Exiting nosmall - n=",n
c           write(*,*) "Debug: lx=",lx,"n=",n,"ijk=",i,j,k,av(i),ksa(i)
                  return
                endif
            endif
            ksp = 0
c           if (mod(i,100).eq.0) write(*,*) "i=",i," a=",a,"n=",
c    &       n,"lzs=",lzs
            nk = lx - i
            do 300 k=0,nk
                cx = float(k) / float(nk+1)
                cf = 1.0 - ( (xmin/x(k+i))**av(i) )
                ks = abs(cf-cx)
                if (ks.gt.ksp)  ksp = ks 
300         continue            
c           write(*,*) "k=",k,"n=",n,"i=",i,"nk=",nk,"j=",j,"cx=",cx
            ksa(i) = ksp
100   continue
c     write(*,*) "Debug: lx=",lx,"n=",n,"ijk=",i,j,k,av(i),ksa(i)
      return
      END
c END FILE fplfit.f
