#ifndef _MESENNE_TWISTER_H_
#define _MESENNE_TWISTER_H_

#define  _USE_MATH_DEFINES
#include <cmath>

/*
	Interface description:

	MTRand( const uint32& oneSeed );  // initialize with a simple uint32
	MTRand( uint32 *const bigSeed, uint32 const seedLength);  // or an array

	// Access to 32-bit random numbers
	double rand();                          // real number in [0,1]
	double rand( const double& n );         // real number in [0,n]
	double randExc();                       // real number in [0,1)
	double randExc( const double& n );      // real number in [0,n)
	double randDblExc();                    // real number in (0,1)
	double randDblExc( const double& n );   // real number in (0,n)
	uint32 randInt();                       // integer in [0,2^32-1]
	uint32 randInt( const uint32& n );      // integer in [0,n] for n < 2^32

	// Access to 53-bit random numbers (capacity of IEEE double precision)
	double rand53();  // real number in [0,1)

	// Access to nonuniform random number distributions
	double randNorm( const double& mean = 0.0, const double& variance = 0.0 );

	// Re-seeding functions with same behavior as initializers
	void seed( const uint32 oneSeed );
	void seed( uint32 *const bigSeed, const uint32 seedLength);

	// return current seed.
	unsigned long seed();

	// Do NOT use for CRYPTOGRAPHY without securely hashing several returned
	// values together, otherwise the generator state can be learned after
	// reading 624 consecutive values.

*/

/** platform dependant RNG. At least faster on x86/amd64 machine.
*/
class MTRandFast
{
	// vc++ 8.0 still sucks by not supporting static constant integer
	// we have to use old #define
	// GCC rocks!

#define MESENNE_TWISTER_N 624
#define MESENNE_TWISTER_M 397
#define MESENNE_TWISTER_MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define MESENNE_TWISTER_MATRIX_UMASK 0x80000000UL /* most significant w-r bits */
#define MESENNE_TWISTER_MATRIX_LMASK 0x7fffffffUL /* least significant r bits */

	typedef unsigned long ulong;

	unsigned long state_[MESENNE_TWISTER_N]; /* the array for the state vector  */
	unsigned long *next_;
	int left_;
	int initf_;
	ulong seed_;

	ulong mixbits(ulong u, ulong v)
	{
		return (u & MESENNE_TWISTER_MATRIX_UMASK) | (v & MESENNE_TWISTER_MATRIX_LMASK);
	}

	ulong twist(ulong u, ulong v)
	{
		return (mixbits(u,v) >> 1) ^ ( ( v & 1UL) ? MESENNE_TWISTER_MATRIX_A : 0UL);
	}


	void next_state(void)
	{
		unsigned long *p=this->state_;
		int j;

		/* if seed() has not been called, */
		/* a default initial seed is used         */
		if (this->initf_==0) seed(5489UL);

		this->left_ = MESENNE_TWISTER_N;
		this->next_ = this->state_;

		for (j=MESENNE_TWISTER_N-MESENNE_TWISTER_M+1; --j; p++)
			*p = p[MESENNE_TWISTER_M] ^ twist(p[0], p[1]);

		for (j=MESENNE_TWISTER_M; --j; p++)
			*p = p[MESENNE_TWISTER_M-MESENNE_TWISTER_N] ^ twist(p[0], p[1]);

		*p = p[MESENNE_TWISTER_M-MESENNE_TWISTER_N] ^ twist(p[0], this->state_[0]);
	}

public:
	MTRandFast()
		: next_(0), left_(1), initf_(0)
	{
	}

	MTRandFast(unsigned long s)
		: next_(0), left_(1), initf_(0)
	{
		this->seed(s);
	}

	MTRandFast(unsigned long init_key[], int key_length)
		: next_(0), left_(1), initf_(0)
	{
		this->seed(init_key, key_length);
	}

	/** initializes this->state_[MESENNE_TWISTER_N] with a seed */
	void seed(unsigned long s)
	{
		int j;
		this->state_[0]= s & 0xffffffffUL;
		for (j=1; j<MESENNE_TWISTER_N; j++) {
			this->state_[j] = (1812433253UL * (this->state_[j-1] ^ (this->state_[j-1] >> 30)) + j);
			/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
			/* In the previous versions, MSBs of the seed affect   */
			/* only MSBs of the array this->state_[].                        */
			/* 2002/01/09 modified by Makoto Matsumoto             */
			this->state_[j] &= 0xffffffffUL;  /* for >32 bit machines */
		}
		this->left_ = 1; this->initf_ = 1;
		this->seed_ = s;
	}

	/** get seed */
	ulong seed(void )
	{
		return this->seed_;
	}

	/** initialize by an array with array-length
	* init_key is the array for initializing keys
	* key_length is its length
	* slight change for C++, 2004/2/26
	*/
	void seed(unsigned long init_key[], int key_length)
	{
		int i, j, k;
		seed(19650218UL);
		i=1; j=0;
		k = (MESENNE_TWISTER_N>key_length ? MESENNE_TWISTER_N : key_length);
		for (; k; k--) {
			this->state_[i] = (this->state_[i] ^ ((this->state_[i-1] ^ (this->state_[i-1] >> 30)) * 1664525UL))
				+ init_key[j] + j; /* non linear */
			this->state_[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
			i++; j++;
			if (i>=MESENNE_TWISTER_N) { this->state_[0] = this->state_[MESENNE_TWISTER_N-1]; i=1; }
			if (j>=key_length) j=0;
		}
		for (k=MESENNE_TWISTER_N-1; k; k--) {
			this->state_[i] = (this->state_[i] ^ ((this->state_[i-1] ^ (this->state_[i-1] >> 30)) * 1566083941UL))
				- i; /* non linear */
			this->state_[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
			i++;
			if (i>=MESENNE_TWISTER_N) { this->state_[0] = this->state_[MESENNE_TWISTER_N-1]; i=1; }
		}

		this->state_[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
		this->left_ = 1; this->initf_ = 1;
	}

    double randNorm( const double& mean, const double& variance )
    {
        // Return a real number from a normal (Gaussian) distribution with given
        // mean and variance by Box-Muller method
        double r = sqrt( -2.0 * log( 1.0-randDblExc()) ) * variance;
        double phi = 2.0 * 3.14159265358979323846264338328 * randExc();
        return mean + r * cos(phi);
    }

    double randNorm()
    {
        // Return a real number from a normal (Gaussian) distribution with given
        // mean and variance by Box-Muller method
        double r = sqrt( -2.0 * log( 1.0-randDblExc()) );
        double phi = 2.0 * 3.14159265358979323846264338328 * randExc();
        return r * cos(phi);
    }

	/// generates a random number on [0,0xffffffff]-interval
	unsigned long randInt(void)
	{
		unsigned long y;

		if (--this->left_ == 0) next_state();
		y = *this->next_++;

		/* Tempering */
		y ^= (y >> 11);
		y ^= (y << 7) & 0x9d2c5680UL;
		y ^= (y << 15) & 0xefc60000UL;
		y ^= (y >> 18);

		return y;
	}

	/// integer in [0,n] for n < 2^32
	ulong randInt( const ulong& n )
	{
		// Find which bits are used in n
		// Optimized by Magnus Jonsson (magnus@smartelectronix.com)
		ulong used = n;
		used |= used >> 1;
		used |= used >> 2;
		used |= used >> 4;
		used |= used >> 8;
		used |= used >> 16;

		// Draw numbers until one is found in [0,n]
		ulong i;
		do
			i = randInt() & used;  // toss unused bits to shorten search
		while( i > n );
		return i;
	}


	/// generates a random number on [0,0x7fffffff]-interval
	long randInt31(void)
	{
		unsigned long y;

		if (--this->left_ == 0) next_state();
		y = *this->next_++;

		// Tempering
		y ^= (y >> 11);
		y ^= (y << 7) & 0x9d2c5680UL;
		y ^= (y << 15) & 0xefc60000UL;
		y ^= (y >> 18);

		return (long)(y>>1);
	}

	/// generates a random number on [0,1]-real-interval
	double rand(void)
	{
		unsigned long y;

		if (--this->left_ == 0) next_state();
		y = *this->next_++;

		// Tempering
		y ^= (y >> 11);
		y ^= (y << 7) & 0x9d2c5680UL;
		y ^= (y << 15) & 0xefc60000UL;
		y ^= (y >> 18);

		return (double)y * (1.0/4294967295.0);
		// divided by 2^32-1
	}

	/// real number in [0,n]
	double rand( const double& n )
	{
		return rand() * n;
	}

	/// generates a random number on [0,1)-real-interval
	double randExc(void)
	{
		unsigned long y;

		if (--this->left_ == 0) next_state();
		y = *this->next_++;

		/* Tempering */
		y ^= (y >> 11);
		y ^= (y << 7) & 0x9d2c5680UL;
		y ^= (y << 15) & 0xefc60000UL;
		y ^= (y >> 18);

		return (double)y * (1.0/4294967296.0);
		/* divided by 2^32 */
	}

	/// real number in [0,n)
	double randExc( const double& n )
	{
		return randExc() * n;
	}

	/// generates a random number on (0,1)-real-interval
	double randDblExc(void)
	{
		unsigned long y;

		if (--this->left_ == 0) next_state();
		y = *this->next_++;

		/* Tempering */
		y ^= (y >> 11);
		y ^= (y << 7) & 0x9d2c5680UL;
		y ^= (y << 15) & 0xefc60000UL;
		y ^= (y >> 18);

		return ((double)y + 0.5) * (1.0/4294967296.0);
		/* divided by 2^32 */
	}

	/// real number in (0,n)
	double randDblExc( const double& n )
	{
		return randDblExc() * n;
	}

	/// generates a random number on [0,1) with 53-bit resolution
	double randRes53(void)
	{
		unsigned long a=randInt()>>5, b=randInt()>>6;
		return(a*67108864.0+b)*(1.0/9007199254740992.0);
	}
	//These real versions are due to Isaku Wada, 2002/01/09 added

};


class MTRand
{
	/* Period parameters */
#define MESENNE_TWISTER_N 624
#define MESENNE_TWISTER_M 397
#define MESENNE_TWISTER_MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define MESENNE_TWISTER_MATRIX_UMASK 0x80000000UL /* most significant w-r bits */
#define MESENNE_TWISTER_MATRIX_LMASK 0x7fffffffUL /* least significant r bits */

	typedef unsigned long ulong;

	unsigned long mt[MESENNE_TWISTER_N]; /* the array for the state vector  */
	int mti; /* mti==MESENNE_TWISTER_N+1 means mt[MESENNE_TWISTER_N] is not initialized */
	ulong seed_;

public:
	MTRand()
		: mti(MESENNE_TWISTER_N+1)
	{
	}

	MTRand(unsigned long s)
		: mti(MESENNE_TWISTER_N+1)
	{
		this->seed(s);
	}

	MTRand(unsigned long init_key[], int key_length)
		: mti(MESENNE_TWISTER_N+1)
	{
		this->seed(init_key, key_length);
	}

	/* initializes mt[MESENNE_TWISTER_N] with a seed */
	void seed(unsigned long s)
	{
		this->seed_ = s;

		mt[0]= s & 0xffffffffUL;
		for (mti=1; mti<MESENNE_TWISTER_N; mti++) {
			mt[mti] =
				(1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
			/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
			/* In the previous versions, MSBs of the seed affect   */
			/* only MSBs of the array mt[].                        */
			/* 2002/01/09 modified by Makoto Matsumoto             */
			mt[mti] &= 0xffffffffUL;
			/* for >32 bit machines */
		}
	}

	ulong seed(void )
	{
		return this->seed_;
	}

	/* initialize by an array with array-length */
	/* init_key is the array for initializing keys */
	/* key_length is its length */
	/* slight change for C++, 2004/2/26 */
	void seed(unsigned long init_key[], int key_length)
	{
		int i, j, k;
		seed(19650218UL);
		i=1; j=0;
		k = (MESENNE_TWISTER_N>key_length ? MESENNE_TWISTER_N : key_length);
		for (; k; k--) {
			mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
				+ init_key[j] + j; /* non linear */
			mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
			i++; j++;
			if (i>=MESENNE_TWISTER_N) { mt[0] = mt[MESENNE_TWISTER_N-1]; i=1; }
			if (j>=key_length) j=0;
		}
		for (k=MESENNE_TWISTER_N-1; k; k--) {
			mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
				- i; /* non linear */
			mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
			i++;
			if (i>=MESENNE_TWISTER_N) { mt[0] = mt[MESENNE_TWISTER_N-1]; i=1; }
		}

		mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
	}

    double randNorm( const double& mean, const double& variance )
    {
        // Return a real number from a normal (Gaussian) distribution with given
        // mean and variance by Box-Muller method
        double r = sqrt( -2.0 * log( 1.0-randDblExc()) ) * variance;
        double phi = 2.0 * 3.14159265358979323846264338328 * randExc();
        return mean + r * cos(phi);
    }

    double randNorm()
    {
        double r = sqrt( -2.0 * log( 1.0-randDblExc()) );
        double phi = 2.0 * 3.14159265358979323846264338328 * randExc();
        return r * cos(phi);
    }

	/* generates a random number on [0,0xffffffff]-interval */
	unsigned long randInt(void)
	{
		unsigned long y;
		static const unsigned long mag01[2]={0x0UL, MESENNE_TWISTER_MATRIX_A};
		/* mag01[x] = x * MESENNE_TWISTER_MATRIX_A  for x=0,1 */

		if (mti >= MESENNE_TWISTER_N) { /* generate MESENNE_TWISTER_N words at one time */
			int kk;

			if (mti == MESENNE_TWISTER_N+1)   /* if seed() has not been called, */
				seed(5489UL); /* a default initial seed is used */

			for (kk=0;kk<MESENNE_TWISTER_N-MESENNE_TWISTER_M;kk++) {
				y = (mt[kk]&MESENNE_TWISTER_MATRIX_UMASK)|(mt[kk+1]&MESENNE_TWISTER_MATRIX_LMASK);
				mt[kk] = mt[kk+MESENNE_TWISTER_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
			}
			for (;kk<MESENNE_TWISTER_N-1;kk++) {
				y = (mt[kk]&MESENNE_TWISTER_MATRIX_UMASK)|(mt[kk+1]&MESENNE_TWISTER_MATRIX_LMASK);
				mt[kk] = mt[kk+(MESENNE_TWISTER_M-MESENNE_TWISTER_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
			}
			y = (mt[MESENNE_TWISTER_N-1]&MESENNE_TWISTER_MATRIX_UMASK)|(mt[0]&MESENNE_TWISTER_MATRIX_LMASK);
			mt[MESENNE_TWISTER_N-1] = mt[MESENNE_TWISTER_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

			mti = 0;
		}

		y = mt[mti++];

		/* Tempering */
		y ^= (y >> 11);
		y ^= (y << 7) & 0x9d2c5680UL;
		y ^= (y << 15) & 0xefc60000UL;
		y ^= (y >> 18);

		return y;
	}

	ulong randInt( const ulong& n )
	{
		// Find which bits are used in n
		// Optimized by Magnus Jonsson (magnus@smartelectronix.com)
		ulong used = n;
		used |= used >> 1;
		used |= used >> 2;
		used |= used >> 4;
		used |= used >> 8;
		used |= used >> 16;

		// Draw numbers until one is found in [0,n]
		ulong i;
		do
			i = randInt() & used;  // toss unused bits to shorten search
		while( i > n );
		return i;
	}

	/* generates a random number on [0,0x7fffffff]-interval */
	long randInt31(void)
	{
		return (long)(randInt()>>1);
	}

	/* generates a random number on [0,1]-real-interval */
	double rand(void)
	{
		return randInt()*(1.0/4294967295.0);
		/* divided by 2^32-1 */
	}

	double rand( const double& n )
	{
		return rand() * n;
	}

	/* generates a random number on [0,1)-real-interval */
	double randExc(void)
	{
		return randInt()*(1.0/4294967296.0);
		/* divided by 2^32 */
	}

	double randExc( const double& n )
	{
		return randExc() * n;
	}

	/* generates a random number on (0,1)-real-interval */
	double randDblExc(void)
	{
		return (((double)randInt()) + 0.5)*(1.0/4294967296.0);
		/* divided by 2^32 */
	}

	double randDblExc( const double& n )
	{
		return randDblExc() * n;
	}

	/* generates a random number on [0,1) with 53-bit resolution*/
	double randRes53(void)
	{
		unsigned long a=randInt()>>5, b=randInt()>>6;
		return(a*67108864.0+b)*(1.0/9007199254740992.0);
	}
	/* These real versions are due to Isaku Wada, 2002/01/09 added */
};



#endif
